"""Collect the outputs of one run of scripts/run_zh_pipeline.sh into a single
JSON/markdown report: greedy (no/full context) scores, MBR outputs, contrastive
decoding outputs and P-CXMI. chrF/BLEU use sacreBLEU (zh tokenizer for Chinese
targets); COMET columns are included when they were computed.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF


def lexical(hyps, refs, tgt):
    hyps = ["" if (isinstance(h, float) and np.isnan(h)) else str(h) for h in hyps]
    return {
        "chrf": round(CHRF().corpus_score(hyps, [list(refs)]).score, 2),
        "bleu": round(BLEU(tokenize="zh" if tgt == "zh" else "13a").corpus_score(hyps, [list(refs)]).score, 2),
        "n": len(hyps),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", required=True)
    p.add_argument("--data_name", default="bmeld")
    p.add_argument("--split", default="test")
    p.add_argument("--model_name", required=True)
    p.add_argument("--gen_backend", default="hf")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    root = Path(args.root_dir)
    ds = f"{args.data_name}_{args.split}"
    report = {"dataset": ds, "model": args.model_name, "greedy": {}, "mbr": {}, "contrastive": {}, "pcxmi": None}

    for ev in sorted(glob.glob(str(root / "evaluations" / "*" / "mt" / f"{ds}.*" / args.gen_backend / args.model_name / "evaluation.json"))):
        parts = Path(ev).parts
        cond, lp = parts[-6], parts[-4].split(".")[-1]
        d = json.load(open(ev))
        report["greedy"][f"{cond}/{lp}"] = {k: v for k, v in d.items() if not k.endswith("_segments")}

    for csv in sorted(glob.glob(str(root / "mbr_outputs" / "**" / "*.csv"), recursive=True)):
        df = pd.read_csv(csv, keep_default_na=False, na_values=["NaN"])
        name = os.path.relpath(csv, root / "mbr_outputs")
        entry = {}
        for lp, g in df.groupby("lp"):
            tgt = lp.split("-")[1]
            entry[lp] = {"mbr": lexical(g["output-select"], g["reference"], tgt)}
            if "greedy" in g:
                entry[lp]["greedy"] = lexical(g["greedy"], g["reference"], tgt)
            if "output-select-comet" in g and g["output-select-comet"].notna().any():
                entry[lp]["mbr"]["comet"] = round(float(g["output-select-comet"].mean()), 4)
        report["mbr"][name] = entry

    data_csv = root / "paper_results_zh" / args.data_name / f"{args.split}.en-zh.csv"
    ref_df = pd.read_csv(data_csv, keep_default_na=False, na_values=["NaN"])
    for out in sorted(glob.glob(str(root / "contrast_decode" / f"{ds}.en-zh" / "*.out.txt"))):
        with open(out) as f:
            hyps = [l.rstrip("\n") for l in f]
        name = Path(out).name.replace(".out.txt", "")
        if len(hyps) != len(ref_df):
            report["contrastive"][name] = {"n_outputs": len(hyps), "note": "sampled (num_return_sequences > 1)"}
            continue
        entry = {}
        for lp, g in ref_df.assign(hyp=hyps).groupby("lp"):
            entry[lp] = lexical(g["hyp"], g["reference"], lp.split("-")[1])
        score_file = out.replace(".out.txt", ".score.txt")
        if os.path.exists(score_file):
            entry["score_txt"] = open(score_file).read().strip().splitlines()
        report["contrastive"][name] = entry

    pc = root / "pcxmi_summary.json"
    if pc.exists():
        report["pcxmi"] = json.load(open(pc))

    out = args.out or root / "pipeline_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
