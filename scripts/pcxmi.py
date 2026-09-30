"""Per-token log-probabilities used for P-CXMI.

P-CXMI (notebooks/pcxmi_analysis.ipynb) for a segment is

    mean_log_prob(y | full_context prompt) - mean_log_prob(y | no_context prompt)

where y is the reference ("ref"), the model's own greedy hypothesis ("hyp")
or, as a control, the source ("src"). This script computes the log-probs for
each (dataset, lp, context setting) and writes

    pcxmi/<context_setting>/mt/<dataset>.<lp>/<model_stem>/{log_probs,mean_log_probs,max_log_probs,min_log_probs,sum_log_probs}_<target>.{jsonl,txt}

Without arguments it reproduces the original run (TowerInstruct-7B-v0.2,
wmt24_chat_test, all original pairs, "ref"). en<->zh example:

  python scripts/pcxmi.py --root_dir . --datasets bmeld_test --lps en-zh zh-en \
      --targets ref hyp --ref_source raw_data
"""

import argparse
import itertools
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_mt_utils import (  # noqa: E402
    CODE_LANG_DICT,
    HFBackend,
    assistant_prefix_token_ids,
    find_token_for_gating,
    load_jsonl,
    read_lines,
    split_lp,
    write_logprob_files,
)

ORIGINAL_LPS = [
    "en-de",
    "de-en",
    "en-fr",
    "fr-en",
    "en-ko",
    "ko-en",
    "en-nl",
    "nl-en",
    "en-pt-br",
    "pt-br-en",
]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default="/mnt/data/jpombal/wmt24-chat-translation")
    p.add_argument("--model", default="Unbabel/TowerInstruct-7B-v0.2")
    p.add_argument("--model_stem", default="TowerInstruct-7B-v0.2")
    p.add_argument("--datasets", nargs="+", default=["wmt24_chat_test"])
    p.add_argument("--lps", nargs="+", default=ORIGINAL_LPS)
    p.add_argument("--context_settings", nargs="+", default=["full_context", "no_context"])
    p.add_argument("--targets", nargs="+", default=["ref"], choices=["ref", "hyp", "src"])
    p.add_argument(
        "--ref_source",
        choices=["paper_results", "raw_data"],
        default="paper_results",
        help="paper_results: paper_results/test.<lp>.csv (original); raw_data: raw_data/mt/<dataset>.<lp>/test.jsonl",
    )
    p.add_argument(
        "--sep_tokens",
        choices=["tower", "chatml", "llama3"],
        default="tower",
        help="tower: hard-coded Tower token ids (original); chatml/llama3: derived from the tokenizer",
    )
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--max_tokens", type=int, default=1024)
    return p.parse_args()


def load_references(args, dataset, lp):
    if args.ref_source == "raw_data":
        return [r["ref"] for r in load_jsonl(Path(args.root_dir) / "raw_data" / "mt" / f"{dataset}.{lp}" / "test.jsonl")]
    if "pt-br" not in lp:
        ref_lp = lp if lp[:2] == "en" else lp[3:] + "-" + lp[:2]
    else:
        ref_lp = "en-pt"
    raw_data_df = pd.read_csv(f"{args.root_dir}/paper_results/test.{ref_lp}.csv")
    return raw_data_df[raw_data_df["lp"] == lp]["reference"].tolist()


class Engine:
    """Thin wrapper so the same code works with vLLM and the HF backend."""

    def __init__(self, args):
        self.backend = args.backend
        if args.backend == "vllm":
            from vllm import LLM, SamplingParams

            self.SamplingParams = SamplingParams
            self.model = LLM(args.model)
        else:
            self.model = HFBackend(args.model)
        self.tokenizer = self.model.get_tokenizer()

    def prompt_logprobs(self, tokenized_prompts):
        if self.backend == "vllm":
            sp = self.SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
            output = self.model.generate(prompt_token_ids=tokenized_prompts, sampling_params=sp, use_tqdm=True)
            # in vllm, each prompt token has a dict whose first key is the prompt token id
            return [[None] + [d[list(d.keys())[0]].logprob for d in out.prompt_logprobs[1:]] for out in output]
        return self.model.prompt_logprobs(tokenized_prompts)

    def greedy_logprobs(self, prompts, max_tokens):
        if self.backend == "vllm":
            sp = self.SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=1)
            output = self.model.generate(prompts, sampling_params=sp, use_tqdm=True)
            return [[d[list(d.keys())[0]].logprob for d in out.outputs[0].logprobs] for out in output]
        return [lps for _, lps in self.model.generate_with_logprobs(prompts, max_tokens=max_tokens)]


def main(args):
    engine = Engine(args)
    tokenizer = engine.tokenizer
    root_dir = Path(args.root_dir)
    raw_data_dir = root_dir / "raw_data" / "mt"
    instructions_dir = root_dir / "instructions"

    if args.sep_tokens == "tower":
        sep_tok_seq = (
            [32000, 29871, 13, 32001, 20255, 13]
            if "chat" in args.model
            else [32005, 29871, 13, 32006, 20255, 13]
        )  # tower instruct has a slightly different token sequence for the separator
    else:
        sep_tok_seq = assistant_prefix_token_ids(tokenizer, args.sep_tokens)

    exp_settings = list(itertools.product(args.datasets, args.lps, args.context_settings))
    for ds, lp, cs in exp_settings:
        e = {"dataset": ds, "lp": lp, "context_setting": cs}
        print(e)
        instructions_path = instructions_dir / cs / "mt" / f"{ds}.{lp}" / "instructions.txt"
        # replicate tower-eval data loading
        instructions = read_lines(instructions_path, unescape_newline=True)
        output_dir = root_dir / "pcxmi" / cs / "mt" / f"{ds}.{lp}" / args.model_stem

        if "ref" in args.targets:
            # PCXMI ON REF
            references = load_references(args, ds, lp)
            assert len(references) == len(
                instructions
            ), "Length of instructions and references must be the same; there may be an error upstream."
            tokenized_prompts, instruction_lengths = [], []
            for inst, ref in zip(instructions, references):
                tokenized_prompt = tokenizer.encode(inst + ref)
                instruction_length = find_token_for_gating(tokenized_prompt, sep_tok_seq) + len(sep_tok_seq)
                ref_length = len(tokenized_prompt) - instruction_length
                # If the generation was less than max tokens, then the model finished with EOS; we must add it.
                if ref_length < 1024:
                    tokenized_prompt.append(tokenizer.eos_token_id)
                    ref_length += 1
                tokenized_prompts.append(tokenized_prompt)
                instruction_lengths.append(instruction_length)
            outputs = engine.prompt_logprobs(tokenized_prompts)
            ref_log_probs_list = [lps[inst_len:] for inst_len, lps in zip(instruction_lengths, outputs)]
            print(f"Writing to {output_dir}...")
            write_logprob_files(output_dir, "ref", ref_log_probs_list)

        if "hyp" in args.targets:
            # PCXMI on hyp (log-probs of the model's own greedy translation)
            log_probs_list = engine.greedy_logprobs(instructions, args.max_tokens)
            print(f"Writing to {output_dir}...")
            write_logprob_files(output_dir, "hyp", log_probs_list)

        if "src" in args.targets:
            # PCXMI on src (control: log-probs of the source given the preceding prompt)
            raw_data = load_jsonl(raw_data_dir / f"{ds}.{lp}" / "test.jsonl")
            sources = [r["src"] for r in raw_data]
            assert len(sources) == len(instructions)
            _, tgt = split_lp(lp)
            tgt_lang = CODE_LANG_DICT[tgt]
            tokenized_prompts, instruction_lengths = [], []
            for inst, src in zip(instructions, sources):
                inst_before_src = re.search(
                    rf"(?P<preamble>[\S\s]*){re.escape(src)}\n{tgt_lang}:", inst
                ).group("preamble")
                tok_prompt = tokenizer.encode(inst_before_src)
                tok_source = tokenizer.encode(src, add_special_tokens=False)
                instruction_lengths.append(len(tok_prompt))
                tokenized_prompts.append(tok_prompt + tok_source)
            outputs = engine.prompt_logprobs(tokenized_prompts)
            src_log_probs_list = [lps[inst_len:] for inst_len, lps in zip(instruction_lengths, outputs)]
            print(f"Writing to {output_dir}...")
            write_logprob_files(output_dir, "src", src_log_probs_list)


if __name__ == "__main__":
    main(get_args())
