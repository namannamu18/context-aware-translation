import os
from pathlib import Path

import pandas as pd
import tqdm
from tower_eval.utils import write_lines
from vllm import LLM, SamplingParams


def apply_instruction(src, src_lang, tgt_lang, tokenizer):
    messages = [
        {
            "role": "system",
            "content": "",
        },
        {
            "role": "user",
            "content": f"Translate the following {src_lang} source text to {tgt_lang}:\n{src_lang}: {src}\n{tgt_lang}: ",
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def apply_series_of_instruction(
    src,
    src_lang,
    tgt_lang,
    tokenizer,
    previous_sources,
    previous_directions,
    current_direction,
):
    initial_string = f"Context: "
    for s, d in zip(previous_sources, previous_directions):
        if d == current_direction:
            l = src_lang
        else:
            l = tgt_lang
        initial_string += f"{s}\n"
    final_string = (
        initial_string
        + f"\nTranslate the {src_lang} source text to {tgt_lang}, given the context."
    )
    final_string = final_string + (f"\n{src_lang}: {src}\n{tgt_lang}: ")
    messages = [
        {
            "role": "system",
            "content": "",
        },
        {
            "role": "user",
            "content": final_string,
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


s = SamplingParams(max_tokens=1024, temperature=0.0)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
model = LLM("Unbabel/TowerInstruct-WMT24-Chat-7B")
t = model.get_tokenizer()

code_lang_dict = {
    "en": "English",
    "de": "German",
    # "fr": "French",
    # "pt-br": "Brazilian Portuguese",
    # "ko": "Korean",
    # "nl": "Dutch",
}

dataset_name = "bcontrast_test"

lps = [f"en-{xx}" for xx in code_lang_dict.keys() if xx != "en"]

for lp in lps:
    turn_count = 0
    print(lp)
    if lp == "en-pt-br":
        non_eng = "pt-br"
        eng = "en"
    else:
        eng, non_eng = lp.split("-")

    outputs = []
    df = pd.read_json(
        f"/mnt/data/jpombal/wmt24-chat-translation/raw_data/mt/{dataset_name}.{non_eng}/test.jsonl",
        lines=True,
    )
    register_convo = False
    last_i = 0
    for row_i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        convo_doc_id = row["doc_id"]
        if row_i != len(df) - 1:
            next_doc_id = df.loc[row_i + 1, "doc_id"]
            if convo_doc_id != next_doc_id:
                register_convo = True
                convo_df = df.loc[last_i:row_i].reset_index(drop=True)
                last_i = row_i + 1
        else:
            register_convo = True
            convo_df = df.loc[last_i:].reset_index(drop=True)
        if register_convo:
            for i, row in convo_df.iterrows():
                src_lang = code_lang_dict[row["source_language"]]
                tgt_lang = code_lang_dict[row["target_language"]]
                if i == 0:
                    previous_outputs = []
                    previous_output_orig_lang = []
                    instruction = apply_instruction(
                        row["src"],
                        src_lang,
                        tgt_lang,
                        t,
                    )
                    output = (
                        model.generate(
                            [instruction], sampling_params=s, use_tqdm=False
                        )[0]
                        .outputs[0]
                        .text
                    )
                    outputs.append(output)
                    previous_outputs.append(output)
                    previous_output_orig_lang.append(src_lang)
                else:
                    previous_sources = convo_df.loc[: i - 1, "src"].tolist()
                    previous_directions = convo_df.loc[: i - 1, "sender"].tolist()
                    # translate sequentially.
                    assert len(previous_sources) == len(previous_outputs)
                    for j, o in enumerate(previous_outputs):
                        if previous_output_orig_lang[j] != "English":
                            previous_sources[j] = o
                    instruction = apply_series_of_instruction(
                        row["src"],
                        src_lang,
                        tgt_lang,
                        t,
                        previous_sources=previous_sources,
                        previous_directions=previous_directions,
                        current_direction=row["sender"],
                    )
                    output = (
                        model.generate(
                            [instruction], sampling_params=s, use_tqdm=False
                        )[0]
                        .outputs[0]
                        .text
                    )
                    outputs.append(output)
                    previous_outputs.append(output)
                    previous_output_orig_lang.append(src_lang)
                    turn_count += len(previous_sources)
        register_convo = False
    print(f"mean turn count: {turn_count/len(df)}")
    df["outputs_seq_mt"] = outputs
    # en-xx direction
    instructions = df[df["source_language"] == "en"]["outputs_seq_mt"].tolist()
    write_lines(
        f"/mnt/data/jpombal/wmt24-chat-translation/generations/english_context_empty_sys/mt/{dataset_name}.{eng}-{non_eng}/vllm/TowerInstruct-7B-w-chat-empty-sys/generation.txt",
        instructions,
        escape_newline=True,
        verbose=False,
    )
    # xx-en direction
    instructions = df[df["target_language"] == "en"]["outputs_seq_mt"].tolist()
    write_lines(
        f"/mnt/data/jpombal/wmt24-chat-translation/generations/english_context_empty_sys/mt/{dataset_name}.{non_eng}-{eng}/vllm/TowerInstruct-7B-w-chat-empty-sys/generation.txt",
        instructions,
        escape_newline=True,
        verbose=False,
    )
