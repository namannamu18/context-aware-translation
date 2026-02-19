from pathlib import Path

import datasets
import pandas as pd


def apply_instruction(src, src_lang, tgt_lang):
    return f"Translate the following {src_lang} source text to {tgt_lang}:\n{src_lang}: {src}\n{tgt_lang}: "


dataset_name = "wmt24_chat_train"

code_lang_dict = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "pt-br": "Brazilian Portuguese",
    "ko": "Korean",
    "nl": "Dutch",
}

lps = [f"en-{xx}" for xx in code_lang_dict.keys() if xx != "en"]

dfs = []
for lp in lps:
    print(lp)
    if lp == "en-pt-br":
        non_eng = "pt-br"
        eng = "en"
    else:
        eng, non_eng = lp.split("-")

    instructions = []
    df = pd.read_json(
        f"/mnt/data/jpombal/wmt24-chat-translation/raw_data/mt/{dataset_name}.{non_eng}/test.jsonl",
        lines=True,
    )
    instructions = (
        df["src"]
        .apply(
            lambda x: apply_instruction(x, code_lang_dict[eng], code_lang_dict[non_eng])
        )
        .tolist()
    )

    df["instructions"] = instructions
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
blocks = datasets.load_dataset("Unbabel/TowerBlocks-v0.2")
blocks_df = blocks["train"].to_pandas()

intermediate_df = pd.DataFrame()
intermediate_df["conversations"] = df.apply(
    lambda x: [
        {"from": "human", "value": x["instructions"]},
        {"from": "gpt", "value": x["ref"]},
    ],
    axis=1,
)
intermediate_df["lang"] = df.apply(
    lambda x: f'{x["source_language"]}-{x["target_language"]}', axis=1
)
intermediate_df["split"] = "train"
intermediate_df["dataset"] = "wmt24_chat_train"
intermediate_df["task"] = "chat_translation"

out_df = pd.concat([blocks_df, intermediate_df], ignore_index=True)
a = 1

from datasets import Dataset

dataset = Dataset.from_pandas(out_df)
dataset.push_to_hub("Unbabel/TowerBlocks-v0.2_w_chat_mt_no_context", private=True)
