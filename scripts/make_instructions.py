"""Build tower-eval instruction files (instructions/<condition>/mt/<dataset>.<lp>/instructions.txt)
for any dataset/language pair, reproducing the prompt formats of
`notebooks/preprocess_data_wmt24_*.ipynb` and `notebooks/make_few_shot_dev_instructions.ipynb`.

Conditions:
  no_context, full_context                       (TowerInstruct chat template)
  no_context_empty_sys, full_context_empty_sys   (chat template w/ empty system turn; TowerChat models)
  full_context_{n}_turns, full_context_empty_sys_{n}_turns   (context truncated to last n utterances)
  no_context_empty_sys_llama3, full_context_empty_sys_llama3 (Tower-Llama3-70B)
  no_context_no_template, full_context_no_template           (no chat template)
  no_context_5_shot, full_context_5_shot         (5 examples drawn from --fewshot_dataset)

Examples:
  # en<->zh, BMELD test, all default conditions
  python scripts/make_instructions.py --datasets bmeld_test bmeld_dev synthetic_chat_test --pairs en-zh
  # regression check against the committed en-de instructions
  python scripts/make_instructions.py --verify
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_mt_utils import (  # noqa: E402
    CODE_LANG_DICT,
    TEMPLATES,
    build_prompts,
    context_content,
    iter_conversations,
    load_jsonl,
    no_context_content,
    read_lines,
    write_lines,
)

N_TURNS = [2, 6, 10, 15, 20]
DEFAULT_CONDITIONS = (
    ["no_context", "full_context", "no_context_empty_sys", "full_context_empty_sys"]
    + [f"full_context_{n}_turns" for n in N_TURNS]
    + [f"full_context_empty_sys_{n}_turns" for n in N_TURNS]
    + [
        "no_context_empty_sys_llama3",
        "full_context_empty_sys_llama3",
        "no_context_no_template",
        "full_context_no_template",
    ]
)


def parse_condition(cond: str):
    """-> (template, use_context, n_turns)"""
    use_context = cond.startswith("full_context")
    n_turns = None
    if cond.endswith("_turns"):
        n_turns = int(cond.split("_")[-2])
    if "llama3" in cond:
        template = "llama3_empty_sys"
    elif "no_template" in cond:
        template = "no_template"
    elif "empty_sys" in cond:
        template = "chatml_empty_sys"
    else:
        template = "chatml"
    return template, use_context, n_turns


def bilingual_records(raw_dir: Path, dataset: str, xx: str):
    """Conversation-ordered file with both directions (<dataset>.<xx>), falling
    back to the blind test file as in the notebooks."""
    for name in [f"{dataset}.{xx}", f"{dataset}_blind.{xx}"]:
        p = raw_dir / name / "test.jsonl"
        if p.exists():
            return load_jsonl(p)
    raise FileNotFoundError(f"No bilingual raw data for {dataset}.{xx} in {raw_dir}")


def split_directions(records, prompts, xx):
    en_xx = [p for r, p in zip(records, prompts) if r["source_language"] == "en"]
    xx_en = [p for r, p in zip(records, prompts) if r["target_language"] == "en"]
    return {f"en-{xx}": en_xx, f"{xx}-en": xx_en}


# ---------------------------------------------------------------------------
# 5-shot prompts (notebooks/make_few_shot_dev_instructions.ipynb)
# ---------------------------------------------------------------------------
def five_shot_no_context(records, xx, fewshot_dir_records):
    out = {}
    for lp in [f"en-{xx}", f"{xx}-en"]:
        s, t = lp.split("-", 1) if not lp.startswith("pt-br") else ("pt-br", "en")
        dev_df = pd.DataFrame(fewshot_dir_records[lp])
        five_examples = dev_df.sample(5, random_state=42).to_dict(orient="records")
        src_lang, tgt_lang = CODE_LANG_DICT[s], CODE_LANG_DICT[t]
        examples_string = "Consider the following examples: "
        for ex in five_examples:
            examples_string += f"{src_lang}: {ex['src']}\n{tgt_lang}: {ex['ref']}\n"
        out[lp] = [
            f"<|im_start|>user\n{examples_string}\nTranslate the following {src_lang} source text to {tgt_lang}:\n{src_lang}: {r['src']}\n{tgt_lang}: <|im_end|>\n<|im_start|>assistant\n"
            for r in records
            if f"{r['source_language']}-{r['target_language']}" == lp
        ]
    return out


def five_shot_example_strings(lps, fewshot_instructions, fewshot_refs):
    np.random.seed(42)
    example_strings = {}
    for lp in lps:
        inputs, refs = fewshot_instructions[lp], fewshot_refs[lp]
        idx = [i for i in range(len(inputs)) if "Context:" in inputs[i]]
        sample = np.random.choice(idx, size=5)
        example_strings[lp] = "Consider the following examples:\n"
        for i in sample:
            example_strings[lp] += f"{inputs[i]}{refs[i]}\n\n"
    return example_strings


def five_shot_full_context(records, example_strings):
    prompts = []
    for convo in iter_conversations(records):
        for i, row in enumerate(convo):
            lp = f"{row['source_language']}-{row['target_language']}"
            src_lang = CODE_LANG_DICT[row["source_language"]]
            tgt_lang = CODE_LANG_DICT[row["target_language"]]
            if i == 0:
                prompts.append(
                    f"<|im_start|>user\n{example_strings[lp]}\nTranslate the following {src_lang} source text to {tgt_lang}:\n{src_lang}: {row['src']}\n{tgt_lang}: <|im_end|>\n<|im_start|>assistant\n"
                )
            else:
                final_string = context_content(row["src"], src_lang, tgt_lang, [r["src"] for r in convo[:i]])
                prompts.append(
                    f"<|im_start|>user\nConsider the following examples: {example_strings[lp]}{final_string}<|im_end|>\n<|im_start|>assistant\n"
                )
    return prompts


def build(root: Path, datasets, pairs, conditions, fewshot_dataset, out_root=None):
    raw_dir = root / "raw_data" / "mt"
    inst_root = (out_root or root) / "instructions"
    for dataset in datasets:
        for pair in pairs:
            xx = pair.split("-", 1)[1]  # en-zh -> zh
            records = bilingual_records(raw_dir, dataset, xx)
            for cond in conditions:
                if "5_shot" in cond:
                    lps = [f"en-{xx}", f"{xx}-en"]
                    if cond == "no_context_5_shot":
                        fs = {lp: load_jsonl(raw_dir / f"{fewshot_dataset}.{lp}" / "test.jsonl") for lp in lps}
                        per_dir = five_shot_no_context(records, xx, fs)
                    else:
                        fs_inst = {
                            lp: read_lines(inst_root / "full_context" / "mt" / f"{fewshot_dataset}.{lp}" / "instructions.txt")
                            for lp in lps
                        }
                        fs_refs = {lp: [r["ref"] for r in load_jsonl(raw_dir / f"{fewshot_dataset}.{lp}" / "test.jsonl")] for lp in lps}
                        ex = five_shot_example_strings(lps, fs_inst, fs_refs)
                        per_dir = split_directions(records, five_shot_full_context(records, ex), xx)
                else:
                    template, use_context, n_turns = parse_condition(cond)
                    prompts = build_prompts(records, template, use_context, n_turns)
                    per_dir = split_directions(records, prompts, xx)
                for lp, prompts in per_dir.items():
                    write_lines(
                        inst_root / cond / "mt" / f"{dataset}.{lp}" / "instructions.txt",
                        prompts,
                        escape_newline=True,
                        verbose=False,
                    )
                print(f"[{cond}] {dataset}: " + ", ".join(f"{lp}={len(v)}" for lp, v in per_dir.items()))


def verify(root: Path, tmp: Path):
    """Regenerate instructions for existing language pairs and compare them
    byte-by-byte with the committed files."""
    checks = [
        ("wmt24_chat_test", "en-de", "no_context"),
        ("wmt24_chat_test", "en-de", "full_context"),
        ("wmt24_chat_test", "en-ko", "full_context_empty_sys"),
        ("wmt24_chat_test", "en-fr", "full_context_empty_sys_6_turns"),
        ("wmt24_chat_test", "en-nl", "full_context_2_turns"),
        ("wmt24_chat_dev", "en-de", "no_context_empty_sys"),
        ("wmt24_chat_dev", "en-pt-br", "full_context_empty_sys_llama3"),
        ("wmt24_chat_test", "en-de", "full_context_no_template"),
        ("bcontrast_test", "en-de", "full_context_empty_sys"),
    ]
    ok = True
    for dataset, pair, cond in checks:
        build(root, [dataset], [pair], [cond], None, out_root=tmp)
        xx = pair.split("-", 1)[1]
        for lp in [pair, f"{xx}-en"]:
            ref = root / "instructions" / cond / "mt" / f"{dataset}.{lp}" / "instructions.txt"
            new = tmp / "instructions" / cond / "mt" / f"{dataset}.{lp}" / "instructions.txt"
            if not ref.exists():
                print(f"  (skip, no reference file) {ref}")
                continue
            a, b = read_lines(ref), read_lines(new)
            same = a == b
            ok &= same
            n_diff = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
            print(f"  {'OK  ' if same else 'DIFF'} {cond} {dataset}.{lp}: {len(b)} lines, {n_diff} differing")
    print("VERIFY:", "PASSED" if ok else "FAILED")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default=".")
    p.add_argument("--out_root", default=None, help="Write instructions under this root instead of --root_dir")
    p.add_argument("--datasets", nargs="+", default=["bmeld_test", "bmeld_dev", "synthetic_chat_test"])
    p.add_argument("--pairs", nargs="+", default=["en-zh"], help="English-centric pairs, e.g. en-zh (both directions are written)")
    p.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    p.add_argument("--fewshot_dataset", default="bmeld_train", help="Dataset the 5-shot examples are drawn from")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    root = Path(args.root_dir)
    if args.verify:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            sys.exit(0 if verify(root, Path(tmp)) else 1)
    build(root, args.datasets, args.pairs, args.conditions, args.fewshot_dataset, Path(args.out_root) if args.out_root else None)


if __name__ == "__main__":
    main()
