import itertools
import json
from pathlib import Path

from tower_eval.utils import read_lines
from vllm import LLM, SamplingParams


def find_token_for_gating(lst, token_pattern):
    """Find the last occurrence of a token_pattern in a list."""
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    for j in range(search_end - token_pattern_len, -1, -1):
        if lst[j : j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")


gen_model_name = "TowerInstruct-7B-w-chat-empty-sys-no-context"
model_stem = "TowerInstruct-7B-w-chat-empty-sys-no-context"
model_name = "Unbabel/TowerInstruct-v0.2-w-chat-mt-data-no-context"
sampling_params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
model = LLM(model_name)
tokenizer = model.get_tokenizer()

root_dir = Path(f"/mnt/data/jpombal/wmt24-chat-translation")
raw_data_dir = root_dir / "raw_data" / "mt"
instructions_dir = root_dir / "instructions"
generations_dir = root_dir / "generations"
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
# run the combination of settings
exp_settings = list(itertools.product(datasets, lps))
exp_settings_records = [{"dataset": ds, "lp": lp} for ds, lp in exp_settings]
full_context_prompt_name = "full_context_empty_sys"
no_context_prompt_name = "no_context_empty_sys"
sep_tok_seq = (
    [32000, 29871, 13, 32001, 20255, 13]
    if model_name != "Unbabel/TowerInstruct-7B-v0.2"
    else [32005, 29871, 13, 32006, 20255, 13]
)  # tower instruct has a slightly different token sequence for the separator
# PCXMI variants
for e in exp_settings_records:
    print(e)
    full_context_instructions_path = (
        instructions_dir
        / full_context_prompt_name
        / "mt"
        / f'{e["dataset"]}.{e["lp"]}'
        / "instructions.txt"
    )
    no_context_instructions_path = (
        instructions_dir
        / no_context_prompt_name
        / "mt"
        / f'{e["dataset"]}.{e["lp"]}'
        / "instructions.txt"
    )
    # replicate tower-eval data loading
    full_context_instructions = read_lines(
        full_context_instructions_path, unescape_newline=True
    )
    no_context_instructions = read_lines(
        no_context_instructions_path, unescape_newline=True
    )

    if "pt-br" not in e["lp"]:
        if e["lp"][:2] == "en":
            ref_lp = e["lp"]
        else:
            ref_lp = e["lp"][3:] + "-" + e["lp"][:2]
    else:
        ref_lp = "en-pt"
    full_context_generations = read_lines(
        f'{generations_dir}/{full_context_prompt_name}/mt/{e["dataset"]}.{e["lp"]}/vllm/{gen_model_name}/generation.txt',
    )
    no_context_generations = read_lines(
        f'{generations_dir}/{no_context_prompt_name}/mt/{e["dataset"]}.{e["lp"]}/vllm/{gen_model_name}/generation.txt',
    )
    assert (
        len(no_context_generations)
        == len(full_context_generations)
        == len(full_context_instructions)
        == len(no_context_instructions)
    ), f"Length of instructions and references must be the same; there may be an error upstream."
    exp_names = ["full_context_input", "no_context_input"]
    ref_exp_names = ["full_context_output", "no_context_output"]
    for instructions, exp_name in zip(
        [full_context_instructions, no_context_instructions], exp_names
    ):
        for references, ref_exp_name in zip(
            [full_context_generations, no_context_generations], ref_exp_names
        ):
            print(f"Generating PCXMI for {exp_name} on {ref_exp_name}...")
            # tokenize prompts and references, and join
            tokenized_prompts = []
            instruction_lengths = []
            for inst, ref in zip(instructions, references):
                if model_name != "Unbabel/TowerInstruct-7B-v0.2":
                    tokenized_prompt = tokenizer.encode(inst + ref)
                    instruction_length = find_token_for_gating(
                        tokenized_prompt, sep_tok_seq
                    ) + len(sep_tok_seq)
                    ref_length = len(tokenized_prompt) - instruction_length
                else:  # handle differently for towerinstruct
                    tok_prompt = tokenizer.encode(inst)
                    instruction_length = len(tok_prompt)
                    tok_reference = tokenizer.encode(
                        ref,
                        add_special_tokens=False,  # do not add bos token because we will merge with prompt; tower instruct doesn't generate a bos token after the translation prompt
                    )
                    ref_length = len(tok_reference)
                    tokenized_prompt = tok_prompt + tok_reference
                # If the generation was less than max tokens, then the model finished with EOS; we must add it. Otherwise, it stopped because of max_tokens condition, and there should be no EOS.
                if ref_length < 1024:
                    tokenized_prompt.append(tokenizer.eos_token_id)
                    ref_length += 1
                tokenized_prompts.append(tokenized_prompt)
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
            # Save information we care about
            output_dir = (
                root_dir
                / "pcxmi_hyps"
                / f"{exp_name}_on_{ref_exp_name}"
                / "mt"
                / f'{e["dataset"].replace("_blind", "")}.{e["lp"]}'
                / model_stem
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Writing to {output_dir}...")
            with open(output_dir / "log_probs.jsonl", "w") as f_jsonl, open(
                output_dir / "mean_log_probs.txt", "w"
            ) as f_mean, open(output_dir / "max_log_probs.txt", "w") as f_max, open(
                output_dir / "min_log_probs.txt", "w"
            ) as f_min, open(
                output_dir / "sum_log_probs.txt", "w"
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
