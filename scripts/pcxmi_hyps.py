"""Cross-condition P-CXMI: log-probs of the full-context and no-context
*generations* under both the full-context and the no-context prompts.

Writes pcxmi_hyps/<{full,no}_context_input>_on_<{full,no}_context_output>/mt/<dataset>.<lp>/<model_stem>/{log_probs.jsonl,mean_log_probs.txt,...}

Without arguments it reproduces the original run
(TowerInstruct-v0.2-w-chat-mt-data-no-context, wmt24_chat_test, all original
pairs). en<->zh example:

  python scripts/pcxmi_hyps.py --root_dir . --datasets bmeld_test --lps en-zh zh-en \
      --model <path> --model_stem <name> --gen_model_name <name> \
      --full_context_prompt_name full_context --no_context_prompt_name no_context \
      --sep_tokens chatml --gen_backend hf --backend hf
"""

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_mt_utils import (  # noqa: E402
    HFBackend,
    assistant_prefix_token_ids,
    find_token_for_gating,
    read_lines,
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
    p.add_argument("--gen_model_name", default="TowerInstruct-7B-w-chat-empty-sys-no-context")
    p.add_argument("--model_stem", default="TowerInstruct-7B-w-chat-empty-sys-no-context")
    p.add_argument("--model", default="Unbabel/TowerInstruct-v0.2-w-chat-mt-data-no-context")
    p.add_argument("--datasets", nargs="+", default=["wmt24_chat_test"])
    p.add_argument("--lps", nargs="+", default=ORIGINAL_LPS)
    p.add_argument("--full_context_prompt_name", default="full_context_empty_sys")
    p.add_argument("--no_context_prompt_name", default="no_context_empty_sys")
    p.add_argument("--sep_tokens", choices=["tower", "chatml", "llama3"], default="tower")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--gen_backend", default="vllm", help="Sub-folder of generations/ holding the hypotheses")
    return p.parse_args()


def main(args):
    model_name = args.model
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams

        sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
        model = LLM(model_name)

        def prompt_logprobs(tokenized_prompts):
            output = model.generate(prompt_token_ids=tokenized_prompts, sampling_params=sampling_params, use_tqdm=True)
            return [[None] + [d[list(d.keys())[0]].logprob for d in out.prompt_logprobs[1:]] for out in output]

    else:
        model = HFBackend(model_name)
        prompt_logprobs = model.prompt_logprobs
    tokenizer = model.get_tokenizer()

    root_dir = Path(args.root_dir)
    instructions_dir = root_dir / "instructions"
    generations_dir = root_dir / "generations"
    if args.sep_tokens == "tower":
        sep_tok_seq = (
            [32000, 29871, 13, 32001, 20255, 13]
            if model_name != "Unbabel/TowerInstruct-7B-v0.2"
            else [32005, 29871, 13, 32006, 20255, 13]
        )  # tower instruct has a slightly different token sequence for the separator
    else:
        sep_tok_seq = assistant_prefix_token_ids(tokenizer, args.sep_tokens)

    for ds, lp in itertools.product(args.datasets, args.lps):
        e = {"dataset": ds, "lp": lp}
        print(e)
        full_context_instructions = read_lines(
            instructions_dir / args.full_context_prompt_name / "mt" / f"{ds}.{lp}" / "instructions.txt", unescape_newline=True
        )
        no_context_instructions = read_lines(
            instructions_dir / args.no_context_prompt_name / "mt" / f"{ds}.{lp}" / "instructions.txt", unescape_newline=True
        )
        full_context_generations = read_lines(
            f"{generations_dir}/{args.full_context_prompt_name}/mt/{ds}.{lp}/{args.gen_backend}/{args.gen_model_name}/generation.txt",
        )
        no_context_generations = read_lines(
            f"{generations_dir}/{args.no_context_prompt_name}/mt/{ds}.{lp}/{args.gen_backend}/{args.gen_model_name}/generation.txt",
        )
        assert (
            len(no_context_generations)
            == len(full_context_generations)
            == len(full_context_instructions)
            == len(no_context_instructions)
        ), "Length of instructions and references must be the same; there may be an error upstream."
        exp_names = ["full_context_input", "no_context_input"]
        ref_exp_names = ["full_context_output", "no_context_output"]
        for instructions, exp_name in zip([full_context_instructions, no_context_instructions], exp_names):
            for references, ref_exp_name in zip([full_context_generations, no_context_generations], ref_exp_names):
                print(f"Generating PCXMI for {exp_name} on {ref_exp_name}...")
                tokenized_prompts, instruction_lengths = [], []
                for inst, ref in zip(instructions, references):
                    if model_name != "Unbabel/TowerInstruct-7B-v0.2":
                        tokenized_prompt = tokenizer.encode(inst + ref)
                        instruction_length = find_token_for_gating(tokenized_prompt, sep_tok_seq) + len(sep_tok_seq)
                        ref_length = len(tokenized_prompt) - instruction_length
                    else:  # handle differently for towerinstruct
                        tok_prompt = tokenizer.encode(inst)
                        instruction_length = len(tok_prompt)
                        tok_reference = tokenizer.encode(ref, add_special_tokens=False)
                        ref_length = len(tok_reference)
                        tokenized_prompt = tok_prompt + tok_reference
                    # If the generation was less than max tokens, then the model finished with EOS; we must add it.
                    if ref_length < 1024:
                        tokenized_prompt.append(tokenizer.eos_token_id)
                        ref_length += 1
                    tokenized_prompts.append(tokenized_prompt)
                    instruction_lengths.append(instruction_length)
                outputs = prompt_logprobs(tokenized_prompts)
                ref_log_probs_list = [lps[inst_len:] for inst_len, lps in zip(instruction_lengths, outputs)]
                output_dir = (
                    root_dir / "pcxmi_hyps" / f"{exp_name}_on_{ref_exp_name}" / "mt" / f'{ds.replace("_blind", "")}.{lp}' / args.model_stem
                )
                print(f"Writing to {output_dir}...")
                write_logprob_files(output_dir, None, ref_log_probs_list)


if __name__ == "__main__":
    main(get_args())
