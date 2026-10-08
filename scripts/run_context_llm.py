# This is largely based on GEMBA-MQM: https://github.com/MicrosoftTranslator/GEMBA/blob/main/gemba_mqm.py

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
from collections import defaultdict

import numpy as np
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


def get_bilingual_context(df, doc_id, seg_id, k, mode="target"):
    """Previous k messages as "Sender (source language): text". mode="target" is the original code: the text is the
    translation (target_seg) of the system being judged; mode="source" shows the original message instead, which is what
    the 1-shot examples of the prompt look like."""
    col = -2 if mode == "target" else 5          # columns: doc_id, segment_id, source_lang, target_lang, sender, source_seg, target_seg, lp
    context_text = []
    for con_seg_id in range(max(0, seg_id - k), seg_id):
        row = df[(df["doc_id"] == doc_id) & (df["segment_id"] == con_seg_id)].values
        assert len(row) == 1
        context_text.append(f"{row[0][4]} ({row[0][2]}): {row[0][col]}")
    return ("\n").join(context_text)


class RateLimiter:
    """At most `rpm` requests per minute over all threads (free API tiers are limited per minute)."""

    def __init__(self, rpm):
        self.interval = 60.0 / rpm if rpm else 0.0
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self):
        if not self.interval:
            return
        with self.lock:
            now = time.time()
            t = max(now, self.next)
            self.next = t + self.interval
        time.sleep(max(0.0, t - now))


class Cache:
    """Answers are appended to a jsonl file, so an interrupted run (quota, session end) continues where it stopped."""

    def __init__(self, path):
        self.path, self.lock, self.data = path, threading.Lock(), {}
        if path and os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                try:
                    d = json.loads(line)
                    self.data[d["key"]] = d["response"]
                except Exception:
                    pass

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "response": value}, ensure_ascii=False) + "\n")


def get_response(client, prompt, model="gpt-4", max_tokens=100, minimal=False, extra_body=None, limiter=None, cache=None, retries=6):
    """One judge request. `minimal` sends only the parameters every OpenAI-compatible server understands (Gemini, vLLM, ...)."""
    key = hashlib.sha1((model + json.dumps(prompt, sort_keys=True, ensure_ascii=False)).encode("utf-8")).hexdigest()
    if cache is not None and cache.get(key) is not None:
        return cache.get(key)
    parameters = {"temperature": 0, "max_tokens": max_tokens, "model": model, "messages": prompt}
    if not minimal:
        parameters.update({"top_p": 1, "n": 1, "frequency_penalty": 0, "presence_penalty": 0, "stop": None})
    if extra_body:
        parameters["extra_body"] = extra_body
    last = None
    for attempt in range(retries):
        if limiter is not None:
            limiter.wait()
        try:
            response = client.chat.completions.create(**parameters)
            output = (response.choices[0].message.content or "").strip()
            if output:
                break
            last = "empty answer"
            parameters["max_tokens"] = min(4096, parameters["max_tokens"] * 4)   # e.g. a reasoning model spent the budget on thinking
        except Exception as e:                                                  # rate limit (429), overload, network
            last = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(min(60, 2 ** attempt * 2))
    else:
        print(f"judge request failed after {retries} attempts ({last})")
        return None
    if cache is not None:
        cache.put(key, output)
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


def normalize_mqm_line(line):
    """Lower-case, drop markdown (**Critical:**, `code`, # headings) and bullets/numbering (- , 1. ), so that the
    answers of different LLMs parse like GPT-4's plain "Critical:" / "accuracy/mistranslation - ..." lines."""
    line = re.sub(r"[*`#_]+", "", line.lower().strip())
    line = re.sub(r"^[\-\u2022\u2013\d\.\)\s]+", "", line)
    return re.sub(r"\s+", " ", line).strip()


NAN_SCORES = lambda: pd.Series([np.nan, np.nan, np.nan, np.nan])  # noqa: E731


def parse_mqm_answer(x, full_desc=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return NAN_SCORES()

    x = str(x)
    if x.startswith('{"improved translation"'):
        print("here")
    errors = {"critical": [], "major": [], "minor": []}
    error_level = None
    saw_header = saw_no_error = False
    for line in x.split("\n"):
        line = normalize_mqm_line(line)
        if not line:
            continue
        header = re.fullmatch(r"(critical|major|minor)( errors?)?\s*:?", line)
        inline = re.fullmatch(r"(critical|major|minor)( errors?)?\s*:\s*(.+)", line)
        if header:
            error_level, saw_header = header.group(1), True
            continue
        if inline and not any(t in inline.group(3) for t in ("no-error", "no error", "no errors")):
            error_level, saw_header = inline.group(1), True
            line = inline.group(3)
        if "no-error" in line or "no error" in line or "no errors" in line:
            saw_no_error = True
            continue
        if error_level is None:
            continue                      # text before the first header (a preface)
        if "non-translation" in line:
            errors["critical"].append(line)
        else:
            errors[error_level].append(line)

    if not saw_header and not saw_no_error:
        print(f"unparsable judge answer: {x[:120]!r}")
        return NAN_SCORES()               # never turn an unreadable answer into a perfect score

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
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai",
                        help="gemini = Google AI Studio's OpenAI-compatible endpoint (free tier), key from GEMINI_API_KEY (or GOOGLE_API_KEY)")
    parser.add_argument("--api_key_env", type=str, default=None, help="name of the environment variable holding the API key")
    parser.add_argument("--rpm", type=float, default=0, help="max requests per minute over all threads (free tiers: ~10; 0 = unlimited)")
    parser.add_argument("--max_tokens", type=int, default=None, help="answer length (default: 100 like GEMBA; 1024 for other endpoints)")
    parser.add_argument("--extra_body_json", type=str, default=None, help='extra request fields as JSON, e.g. \'{"reasoning_effort": "none"}\'')
    parser.add_argument("--cache_file", type=str, default=None, help="jsonl file with finished answers; makes the run resumable")
    parser.add_argument("--max_docs", type=int, default=None, help="judge only the first N conversations (cheaper)")
    parser.add_argument("--context_mode", choices=["target", "source"], default="target",
                        help="target (original code): the context shows the translations of the system being judged; source: the original messages")
    args = parser.parse_args()
    return args


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def make_client(args):
    """Returns (client, minimal_params). The key is read from the environment only and never printed."""
    if args.provider == "gemini":
        args.base_url = args.base_url or GEMINI_BASE_URL
        if args.judge_model == "gpt-4":
            args.judge_model = "gemini-2.5-flash"
        key = os.environ.get(args.api_key_env or "GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise SystemExit("No Gemini API key: set GEMINI_API_KEY (Kaggle: Add-ons -> Secrets).")
        return OpenAI(api_key=key, base_url=args.base_url), True
    if args.base_url is None:
        kwargs = {"api_key": os.environ[args.api_key_env or "OPENAI_API_KEY"]}
        if os.environ.get("OPENAI_API_ORG"):
            kwargs["organization"] = os.environ["OPENAI_API_ORG"]
        return OpenAI(**kwargs), False
    return OpenAI(api_key=os.environ.get(args.api_key_env or "OPENAI_API_KEY", "EMPTY"), base_url=args.base_url), True


def main(args):
    client, minimal = make_client(args)
    max_tokens = args.max_tokens or (1024 if minimal else 100)
    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else None
    limiter = RateLimiter(args.rpm)
    cache = Cache(args.cache_file)
    if not args.fixed_ende_examples and f"en{args.lp}_conversation" not in few_shots_context:
        # no 1-shot conversation example exists for this language pair (e.g. en-zh)
        print(f"### No en{args.lp} few-shot example; falling back to the fixed en-de example ###")
        args.fixed_ende_examples = True
    if args.fixed_ende_examples:
        print("### USING FIXED EN-DE FEWSHOT EXAMPLES ###")

    dfs_all = pd.read_csv(
        f"{args.data_dir}/{args.dataset}/{args.split}.en-{args.lp}.csv", index_col=None, keep_default_na=False, na_values=[]
    )
    if args.max_docs:
        keep = list(dict.fromkeys(dfs_all["doc_id"]))[: args.max_docs]
        dfs_all = dfs_all[dfs_all["doc_id"].isin(keep)].copy()
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
                    dfs_all, row["doc_id"], row["segment_id"], args.context_size, args.context_mode
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
            future = executor.submit(get_response, client, p, args.judge_model, max_tokens, minimal, extra_body, limiter, cache)
            future_to_index[future] = idx

        # Process completed futures
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_index), total=len(prompts)
        ):
            idx = future_to_index[future]
            results[idx] = future.result()

    n_failed = sum(r is None for r in results)
    print(f"judged {len(results) - n_failed}/{len(results)} segments" + (f" ({n_failed} requests failed: re-run to resume from the cache)" if n_failed else ""))
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
