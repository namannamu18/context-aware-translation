import itertools
import json
import re
from pathlib import Path

import pandas as pd
from tower_eval.utils import load_jsonl_file, read_lines
from vllm import LLM, SamplingParams


def find_token_for_gating(lst, token_pattern):
    """Find the last occurrence of a token_pattern in a list."""
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    for j in range(search_end - token_pattern_len, -1, -1):
        if lst[j : j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")


model_stem = "TowerInstruct-7B-v0.2"
model_name = "Unbabel/TowerInstruct-7B-v0.2"
sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
model = LLM(model_name)
tokenizer = model.get_tokenizer()

root_dir = Path(f"/mnt/data/jpombal/wmt24-chat-translation")
raw_data_dir = root_dir / "raw_data" / "mt"
instructions_dir = root_dir / "instructions"
datasets = ["wmt24_chat_test"]
lps = [
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
context_settings = [
    "full_context",
    "no_context",
]
# run the combination of settings
exp_settings = list(itertools.product(datasets, lps, context_settings))
exp_settings_records = [
    {"dataset": ds, "lp": lp, "context_setting": cs} for ds, lp, cs in exp_settings
]

sep_tok_seq = (
    [32000, 29871, 13, 32001, 20255, 13]
    if "chat" in model_name
    else [32005, 29871, 13, 32006, 20255, 13]
)  # tower instruct has a slightly different token sequence for the separator

# PCXMI ON REF
print("PCXMI on ref")
for e in exp_settings_records:
    print(e)
    instructions_path = (
        instructions_dir
        / e["context_setting"]
        / "mt"
        / f'{e["dataset"]}.{e["lp"]}'
        / "instructions.txt"
    )
    # replicate tower-eval data loading
    instructions = read_lines(instructions_path, unescape_newline=True)
    if "pt-br" not in e["lp"]:
        if e["lp"][:2] == "en":
            ref_lp = e["lp"]
        else:
            ref_lp = e["lp"][3:] + "-" + e["lp"][:2]
    else:
        ref_lp = "en-pt"
    raw_data_df = pd.read_csv(
        f"/mnt/data/jpombal/wmt24-chat-translation/paper_results/test.{ref_lp}.csv"
    )
    references = raw_data_df[raw_data_df["lp"] == e["lp"]]["reference"].tolist()
    assert len(references) == len(
        instructions
    ), f"Length of instructions and references must be the same; there may be an error upstream."
    # tokenize prompts and references, and join
    tokenized_prompts = []
    instruction_lengths = []
    for inst, ref in zip(instructions, references):
        # TODO: ADD IF FOR TOWERINSTRUCT; (HANDLE LIKE BEFORE)
        tokenized_prompt = tokenizer.encode(inst + ref)
        tokenized_prompts.append(tokenized_prompt)
        instruction_length = find_token_for_gating(tokenized_prompt, sep_tok_seq) + len(
            sep_tok_seq
        )
        ref_length = len(tokenized_prompt) - instruction_length
        # If the generation was less than max tokens, then the model finished with EOS; we must add it. Otherwise, it stopped because of max_tokens condition, and there should be no EOS.
        if ref_length < 1024:
            tokenized_prompt.append(tokenizer.eos_token_id)
            ref_length += 1
        instruction_lengths.append(
            instruction_length
        )  # register the length of the instruction for later
    output = model.generate(
        prompt_token_ids=tokenized_prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    ref_log_probs_list = []
    for inst_len, out in zip(instruction_lengths, output):
        # get the log probabilities of each token in the reference.
        # in vllm, log probs are in a list, where each token has a dict with at most two values: the first key is the prompt token id, and respective log prob;
        # if that token is not the model's first token, that dict will have a second key, which is the token id of token with the highest log prob;
        # In this case we only care about the former
        ref_log_probs = [
            d[list(d.keys())[0]].logprob for d in out.prompt_logprobs[inst_len:]
        ]
        ref_log_probs_list.append(ref_log_probs)
    breakpoint()
    # Save information we care about
    output_dir = (
        root_dir
        / "pcxmi"
        / e["context_setting"]
        / "mt"
        / f'{e["dataset"]}.{e["lp"]}'
        / model_stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_dir}...")
    with open(output_dir / "log_probs_ref.jsonl", "w") as f_jsonl, open(
        output_dir / "mean_log_probs_ref.txt", "w"
    ) as f_mean, open(output_dir / "max_log_probs_ref.txt", "w") as f_max, open(
        output_dir / "min_log_probs_ref.txt", "w"
    ) as f_min, open(
        output_dir / "sum_log_probs_ref.txt", "w"
    ) as f_sum:
        for ref_log_probs in ref_log_probs_list:
            # coalesce to zero if no tokens in the reference; happens once in nl-en
            if ref_log_probs == []:
                ref_log_probs = [-0.0]
            json.dump({"log_probs": ref_log_probs}, f_jsonl)
            f_jsonl.write("\n")
            f_mean.write(f"{sum(ref_log_probs) / len(ref_log_probs)}\n")
            f_max.write(f"{max(ref_log_probs)}\n")
            f_min.write(f"{min(ref_log_probs)}\n")
            f_sum.write(f"{sum(ref_log_probs)}\n")


# # PCXMI on hyp
"""
sampling_params = SamplingParams(temperature=0.0, max_tokens=1024, logprobs=1)
print("PCXMI on hyp")
for e in exp_settings_records:
    print(e)
    instructions_path = (
        instructions_dir
        / e["context_setting"]
        / "mt"
        / f'{e["dataset"]}.{e["lp"]}'
        / "instructions.txt"
    )
    # replicate tower-eval data loading
    instructions = read_lines(instructions_path, unescape_newline=True)
    output = model.generate(
        instructions,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    log_probs_list = []
    for out in output:
        # get the log probabilities of each token in the output (hypothesis)
        log_probs = [d[list(d.keys())[0]].logprob for d in out.outputs[0].logprobs]
        log_probs_list.append(log_probs)
    # Save information we care about
    output_dir = (
        root_dir
        / "pcxmi"
        / e["context_setting"]
        / "mt"
        / f'{e["dataset"]}.{e["lp"]}'
        / model_stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_dir}...")
    with open(output_dir / "log_probs_hyp.jsonl", "w") as f_jsonl, open(
        output_dir / "mean_log_probs_hyp.txt", "w"
    ) as f_mean, open(output_dir / "max_log_probs_hyp.txt", "w") as f_max, open(
        output_dir / "min_log_probs_hyp.txt", "w"
    ) as f_min, open(
        output_dir / "sum_log_probs_hyp.txt", "w"
    ) as f_sum:
        for log_probs in log_probs_list:
            json.dump({"log_probs": log_probs}, f_jsonl)
            f_jsonl.write("\n")
            f_mean.write(f"{sum(log_probs) / len(log_probs)}\n")
            f_max.write(f"{max(log_probs)}\n")
            f_min.write(f"{min(log_probs)}\n")
            f_sum.write(f"{sum(log_probs)}\n")
"""

"""
# PCXMI on src
sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
code_lang_dict = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "pt-br": "Brazilian Portuguese",
    "ko": "Korean",
    "nl": "Dutch",
}

for e in exp_settings_records:
    print(e)
    instructions_path = (
        instructions_dir
        / e["context_setting"]
        / "mt"
        / f'{e["dataset"]}.{e["lp"]}'
        / "instructions.txt"
    )
    raw_data_path = raw_data_dir / f'{e["dataset"]}.{e["lp"]}' / "test.jsonl"
    # replicate tower-eval data loading
    instructions = read_lines(instructions_path, unescape_newline=True)
    raw_data = load_jsonl_file(raw_data_path)
    sources = [r["src"] for r in raw_data]
    assert len(sources) == len(
        instructions
    ), f"Length of instructions and sources must be the same; there may be an error upstream."
    # tokenize prompts and references, and join
    tokenized_prompts = []
    instruction_lengths = []
    for inst, src in zip(instructions, sources):
        if "br" not in e["lp"]:
            tgt_lang = code_lang_dict[e["lp"].split("-")[-1]]
        else:
            if e["lp"][:2] == "en":
                tgt_lang = "Brazilian Portuguese"
            else:
                tgt_lang = "English"
        inst_before_src = re.search(
            rf"(?P<preamble>[\S\s]*){re.escape(src)}\n{tgt_lang}:",
            inst,
        ).group("preamble")
        tok_prompt = tokenizer.encode(inst_before_src)
        tok_source = tokenizer.encode(
            src,
            add_special_tokens=False,  # do not add bos token because we will merge with prompt; tower instruct doesn't generate a bos token after the translation prompt
        )
        instruction_lengths.append(
            len(tok_prompt)
        )  # register the length of the instruction for later
        tokenized_prompt = tok_prompt + tok_source
        tokenized_prompts.append(tokenized_prompt)
    output = model.generate(
        prompt_token_ids=tokenized_prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    src_log_probs_list = []
    for inst_len, out in zip(instruction_lengths, output):
        # get the log probabilities of each token in the reference.
        # in vllm, log probs are in a list, where each token has a dict with at most two values: the first key is the prompt token id, and respective log prob;
        # if that token is not the model's first token, that dict will have a second key, which is the token id of token with the highest log prob;
        # In this case we only care about the former
        src_log_probs = [
            d[list(d.keys())[0]].logprob for d in out.prompt_logprobs[inst_len:]
        ]
        src_log_probs_list.append(src_log_probs)
    # Save information we care about
    output_dir = (
        root_dir
        / "pcxmi"
        / e["context_setting"]
        / "mt"
        / f'{e["dataset"]}.{e["lp"]}'
        / model_stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_dir}...")
    with open(output_dir / "log_probs_src.jsonl", "w") as f_jsonl, open(
        output_dir / "mean_log_probs_src.txt", "w"
    ) as f_mean, open(output_dir / "max_log_probs_src.txt", "w") as f_max, open(
        output_dir / "min_log_probs_src.txt", "w"
    ) as f_min, open(
        output_dir / "sum_log_probs_src.txt", "w"
    ) as f_sum:
        for src_log_probs in src_log_probs_list:
            # coalesce to zero if no tokens in the source; happens once in nl-en
            if src_log_probs == []:
                src_log_probs = [-0.0]
            json.dump({"log_probs": src_log_probs}, f_jsonl)
            f_jsonl.write("\n")
            f_mean.write(f"{sum(src_log_probs) / len(src_log_probs)}\n")
            f_max.write(f"{max(src_log_probs)}\n")
            f_min.write(f"{min(src_log_probs)}\n")
            f_sum.write(f"{sum(src_log_probs)}\n")
"""
