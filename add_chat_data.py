import datasets
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

root_dir = "/mnt/data/jpombal/wmt24-chat-translation"
dataset_name = "wmt24_chat_train"

t = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-13B-v0.2")


def apply_instruction(src, src_lang, tgt_lang, tokenizer):
    return f"Translate the following {src_lang} source text to {tgt_lang}:\n{src_lang}: {src}\n{tgt_lang}: "


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

    return final_string


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
    df = pd.read_csv(
        f"/mnt/data/jpombal/wmt24-chat-translation/mbr_outputs_train/full_context_empty_sys/TowerInstruct-7B-w-chat-empty-sys/wmt24_chat_train.{lp}/comet_eps_context_source_w2.csv",
    )
    register_convo = False
    last_i = 0
    for row_i, row in df.iterrows():
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
                    instructions.append(
                        apply_instruction(
                            row["source"],
                            src_lang,
                            tgt_lang,
                            t,
                        )
                    )
                else:
                    previous_sources = convo_df.loc[: i - 1, "source"].tolist()
                    previous_directions = convo_df.loc[: i - 1, "sender"].tolist()
                    instructions.append(
                        apply_series_of_instruction(
                            row["source"],
                            src_lang,
                            tgt_lang,
                            t,
                            previous_sources=previous_sources,
                            previous_directions=previous_directions,
                            current_direction=row["sender"],
                        )
                    )
        register_convo = False

    df["instructions"] = instructions
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df.to_json(
    "/mnt/data/jpombal/wmt24-chat-translation/mbr_outputs_train/chat_mt_w_context.jsonl",
    lines=True,
    orient="records",
    force_ascii=False,
)

blocks = datasets.load_dataset("Unbabel/TowerBlocks-v0.2")
blocks_df = blocks["train"].to_pandas()

intermediate_df = pd.DataFrame()
intermediate_df["conversations"] = df.apply(
    lambda x: [
        {"from": "human", "value": x["instructions"]},
        {"from": "gpt", "value": x["output-select"]},
    ],
    axis=1,
)
intermediate_df["lang"] = df.apply(
    lambda x: f'{x["source_language"]}-{x["target_language"]}', axis=1
)
intermediate_df["split"] = "train"
intermediate_df["dataset"] = "wmt24_chat_train_ctx_comet_mbr_distill"
intermediate_df["task"] = "chat_translation"

# save mbr only
dataset = Dataset.from_pandas(intermediate_df)
dataset.push_to_hub("Unbabel/chat_mt_ctx_comet_mbr_distill", private=True)

# save blocks w/ mbr
out_df = pd.concat([blocks_df, intermediate_df], ignore_index=True)

dataset = Dataset.from_pandas(out_df)
dataset.push_to_hub(
    "Unbabel/TowerBlocks-v0.2_w_chat_mt_ctx_comet_mbr_distill", private=True
)
