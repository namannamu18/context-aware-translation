# This is largely based on GEMBA-MQM: https://github.com/MicrosoftTranslator/GEMBA/blob/main/gemba_mqm.py

import argparse
import concurrent.futures
import os
from collections import defaultdict

import pandas as pd
import tqdm
from llm_fewshot_examples import TEMPLATE_GEMBA_CONTEXT_MQM_1shot, few_shots_context
from openai import OpenAI

lang_dict = {
    "en": "English",
    "de": "German",
    "pt-br": "Portuguese",
    "fr": "French",
    "nl": "Dutch",
    "ko": "Korean",
    "zh": "Chinese",
}


def apply_template(template, data):
    if isinstance(template, str):
        return template.format(**data)
    elif isinstance(template, list):
        prompt = []
        for conversation_turn in template:
            p = conversation_turn.copy()
            p["content"] = p["content"].format(**data)
            prompt.append(p)
        return prompt
    else:
        raise ValueError(f"Unknown template type {type(template)}")


def get_bilingual_context(df, doc_id, seg_id, k):
    context_text = []
    for con_seg_id in range(max(0, seg_id - k), seg_id):
        row = df[(df["doc_id"] == doc_id) & (df["segment_id"] == con_seg_id)].values
        assert len(row) == 1
        context_text.append(f"{row[0][4]} ({row[0][2]}): {row[0][-2]}")
    return ("\n").join(context_text)


def get_response(client, prompt, model="gpt-4"):
    parameters = {
        "temperature": 0,
        "max_tokens": 100,
        "top_p": 1,
        "n": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
        "model": model,
        "messages": prompt,
    }
    response = client.chat.completions.create(**parameters)
    output = response.choices[0].message.content.strip()
    print(output)
    return output


def parse_error_class(error):
    # parse error from error description, errors are ['accuracy', 'fluency', 'locale convention', 'style', 'terminology', 'non-translation', 'other']
    #  locale convention (currency, date, name, telephone, or time format), style (awkward), terminology (inappropriate for context, inconsistent use),
    class_name = "unknown"
    if "accuracy" in error:
        class_name = "accuracy"
        for subclass in ["addition", "mistranslation", "omission", "untranslated text"]:
            if subclass in error:
                class_name = f"accuracy-{subclass}"
    elif "fluency" in error:
        class_name = "fluency"
        for subclass in [
            "character encoding",
            "grammar",
            "inconsistency",
            "punctuation",
            "register",
            "spelling",
        ]:
            if subclass in error:
                class_name = f"fluency-{subclass}"
    elif "locale convention" in error:
        class_name = "locale convention"
        for subclass in ["currency", "date", "name", "telephone", "time"]:
            if subclass in error:
                class_name = f"locale convention-{subclass}"
    elif "style" in error:
        class_name = "style"
    elif "terminology" in error:
        class_name = "terminology"
        for subclass in ["inappropriate", "inconsistent"]:
            if subclass in error:
                class_name = f"terminology-{subclass}"
    elif "non-translation" in error:
        class_name = "non-translation"
    elif "other" in error:
        class_name = "other"

    return class_name


def parse_mqm_answer(x, full_desc=True):
    if x is None:
        return None

    x = str(x)
    if x.startswith('{"improved translation"'):
        print("here")
    else:
        x = x.lower()
        errors = {"critical": [], "major": [], "minor": []}
        error_level = None
        for line in x.split("\n"):
            line = line.strip()
            if (
                "no-error" in line
                or "no error" in line
                or "no errors" in line
                or "" == line
            ):
                continue
            if "critical:" == line:
                error_level = "critical"
                continue
            elif "major:" == line:
                error_level = "major"
                continue
            elif "minor:" == line:
                error_level = "minor"
                continue

            if "critical" in line or "major" in line or "minor" in line:
                if not any(
                    [
                        line.startswith(x)
                        for x in [
                            "accuracy",
                            "fluency",
                            "locale convention",
                            "style",
                            "terminology",
                            "non-translation",
                            "other",
                        ]
                    ]
                ):
                    print(line)

            if error_level is None:
                print(f"No error level for {line}")
                continue

            if "non-translation" in line:
                errors["critical"].append(line)
            else:
                errors[error_level].append(line)

    error_classes = defaultdict(list)
    final_score = 0
    error_counter = {"critical": 0, "major": 0, "minor": 0}
    for error_level in ["critical", "major", "minor"]:
        if error_level not in errors:
            continue
        for error in errors[error_level]:
            final_score += (
                10 if error_level == "critical" else 5 if error_level == "major" else 1
            )
            error_counter[error_level] += 1

            if full_desc:
                error_classes[error_level].append(error)
            else:
                class_name = parse_error_class(error)
                error_classes[error_level].append(class_name)

    # We remove this for chat data as human annotations were collected without this constraint unlike other WMT tasks
    # if final_score > 25:
    #     final_score = 25

    return pd.Series(
        [
            -final_score,
            error_counter["critical"],
            error_counter["major"],
            error_counter["minor"],
        ]
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lp", type=str, default="de")
    parser.add_argument("--dataset", type=str, default="wmt24")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="gpt-4")
    parser.add_argument("--tgt_col", default="mqm", type=str)
    parser.add_argument("--context_size", type=int, default=8)
    parser.add_argument(
        "--fixed_ende_examples", action="store_true", help="Use fixed en-de examples"
    )
    # Free / open-weight judges: any OpenAI-compatible server, e.g.
    #   vllm serve Qwen/Qwen2.5-72B-Instruct --port 8000
    #   python run_context_llm.py --base_url http://localhost:8000/v1 --judge_model Qwen/Qwen2.5-72B-Instruct ...
    parser.add_argument("--base_url", type=str, default=None, help="OpenAI-compatible endpoint (default: OpenAI API)")
    parser.add_argument("--judge_model", type=str, default="gpt-4", help="Model name sent to the API")
    parser.add_argument("--data_dir", type=str, default="../tacl_review")
    parser.add_argument("--max_workers", type=int, default=16)
    args = parser.parse_args()
    return args


def main(args):
    if args.base_url is None:
        credentials = {
            "deployments": {args.model_name: args.model_name},
            "api_key": os.environ["OPENAI_API_KEY"],  # add api-key
            "requests_per_second_limit": 1,
            "organization": os.environ["OPENAI_API_ORG"],  # add org-key
        }
        client = OpenAI(
            api_key=credentials["api_key"],
            organization=credentials["organization"],
        )
    else:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), base_url=args.base_url)
    if not args.fixed_ende_examples and f"en{args.lp}_conversation" not in few_shots_context:
        # no 1-shot conversation example exists for this language pair (e.g. en-zh)
        print(f"### No en{args.lp} few-shot example; falling back to the fixed en-de example ###")
        args.fixed_ende_examples = True
    if args.fixed_ende_examples:
        print("### USING FIXED EN-DE FEWSHOT EXAMPLES ###")

    dfs_all = pd.read_csv(
        f"{args.data_dir}/{args.dataset}/{args.split}.en-{args.lp}.csv", index_col=None
    )
    dfs_all["src_len"] = dfs_all["source"].apply(lambda x: len(x.split(" ")))

    dfs_all["sender"] = dfs_all["sender"].replace("agent", "Agent")
    dfs_all["sender"] = dfs_all["sender"].replace("customer", "Customer")

    dfs_all["lp"] = dfs_all["source_language"] + "_" + dfs_all["target_language"]

    all_df = []
    for _, gr_df in dfs_all.groupby("doc_id"):
        gr_df["segment_id"] = list(range(0, len(gr_df)))
        all_df.append(gr_df)
    dfs_all = pd.concat(all_df)

    dfs_all["source_language"] = dfs_all["source_language"].apply(
        lambda x: lang_dict[x]
    )
    dfs_all["target_language"] = dfs_all["target_language"].apply(
        lambda x: lang_dict[x]
    )

    dfs_all.rename(
        columns={
            "source": "source_seg",
            f"{args.tgt_col}": "target_seg",
            "source_language": "source_lang",
            "target_language": "target_lang",
        },
        inplace=True,
    )

    dfs_all = dfs_all[
        [
            "doc_id",
            "segment_id",
            "source_lang",
            "target_lang",
            "sender",
            "source_seg",
            "target_seg",
            "lp",
        ]
    ]

    context = []
    for _, row in dfs_all.iterrows():
        if row["segment_id"] == 0:
            context.append("")
        else:
            context.append(
                get_bilingual_context(
                    dfs_all, row["doc_id"], row["segment_id"], args.context_size
                )
            )

    dfs_all["context"] = context
    if args.fixed_ende_examples:
        examples_l = "de"
        path_str = ".fixed_ende_examples_1shot"
    else:
        examples_l = args.lp
        path_str = ""
    dfs_all["context_prompt"] = dfs_all.apply(
        lambda x: apply_template(
            TEMPLATE_GEMBA_CONTEXT_MQM_1shot(f"en{examples_l}_conversation"), x
        ),
        axis=1,
    )
    # print 3 random context prompts to validate format
    for i in range(3):
        print(dfs_all["context_prompt"].sample().values[0])

    prompts = dfs_all["context_prompt"].tolist()
    results = [None] * len(prompts)  # Pre-allocate the results list
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Create a dictionary to map futures to their indices
        future_to_index = {}

        # Submit all tasks and store their futures with indices
        for idx, p in enumerate(prompts):
            future = executor.submit(get_response, client, p, args.judge_model)
            future_to_index[future] = idx

        # Process completed futures
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_index), total=len(prompts)
        ):
            idx = future_to_index[future]
            results[idx] = future.result()

    dfs_all[f"{args.model_name}-result"] = results
    dfs_all[
        [
            f"{args.model_name}-score",
            f"{args.model_name}-critical-count",
            f"{args.model_name}-major",
            f"{args.model_name}-minor",
        ]
    ] = dfs_all[f"{args.model_name}-result"].apply(parse_mqm_answer)

    dfs_all.to_csv(
        f"{args.data_dir}/{args.dataset}/{args.split}.en-{args.lp}-{args.tgt_col}.gemba-{args.model_name}{path_str}.csv",
        index=None,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
