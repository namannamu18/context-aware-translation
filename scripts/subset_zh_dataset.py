"""Copy the first N conversations of a prepared en<->zh dataset into another
root directory (raw_data/ + paper_results_zh/), e.g. to run the whole pipeline
quickly on CPU. The subset keeps the exact file layout and formats.

  python scripts/subset_zh_dataset.py --dst_root runs/smoke1 --data_name bmeld --split test --max_docs 12
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_root", default=".")
    p.add_argument("--dst_root", required=True)
    p.add_argument("--data_name", default="bmeld")
    p.add_argument("--split", default="test")
    p.add_argument("--max_docs", type=int, default=None)
    p.add_argument("--skip_docs", type=int, default=0, help="Skip the first K conversations (to draw a different subset)")
    args = p.parse_args()
    src, dst = Path(args.src_root), Path(args.dst_root)
    ds = f"{args.data_name}_{args.split}"

    full = pd.read_json(src / "raw_data" / "mt" / f"{ds}.zh" / "test.jsonl", lines=True, dtype=False)
    docs = list(dict.fromkeys(full["doc_id"]))
    docs = docs[args.skip_docs :]
    if args.max_docs:
        docs = docs[: args.max_docs]
    keep = set(docs)
    sub = full[full["doc_id"].isin(keep)].reset_index(drop=True)
    for name, df in [
        (f"{ds}.zh", sub),
        (f"{ds}.en-zh", sub[sub["source_language"] == "en"]),
        (f"{ds}.zh-en", sub[sub["target_language"] == "en"]),
    ]:
        d = dst / "raw_data" / "mt" / name
        d.mkdir(parents=True, exist_ok=True)
        df.reset_index(drop=True).to_json(d / "test.jsonl", orient="records", lines=True, force_ascii=False)
    pr = pd.read_csv(src / "paper_results_zh" / args.data_name / f"{args.split}.en-zh.csv", keep_default_na=False, na_values=["NaN"])
    pr = pr[pr["doc_id"].isin(keep)].reset_index(drop=True)
    d = dst / "paper_results_zh" / args.data_name
    d.mkdir(parents=True, exist_ok=True)
    pr.to_csv(d / f"{args.split}.en-zh.csv", index=False)
    print(f"{ds}: {len(docs)} conversations, {len(sub)} segments -> {dst}")


if __name__ == "__main__":
    main()
