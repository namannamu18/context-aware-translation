"""Aggregate P-CXMI from the files written by scripts/pcxmi.py and
scripts/pcxmi_hyps.py (same definition as notebooks/pcxmi_analysis.ipynb):

    pcxmi_<tgt>[i] = full_context mean_log_prob_<tgt>[i] - no_context mean_log_prob_<tgt>[i]

Also reports, for pcxmi_hyps, the P-CXMI of the full-context and no-context
outputs (full_context_input_on_X - no_context_input_on_X).

Example:
  python scripts/pcxmi_summary.py --root_dir . --datasets bmeld_test --lps en-zh zh-en \
      --model_stem TowerInstruct-7B-v0.2 --out pcxmi_summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def read_floats(p):
    with open(p) as f:
        return np.array([float(l) for l in f if l.strip()])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default=".")
    p.add_argument("--datasets", nargs="+", default=["bmeld_test"])
    p.add_argument("--lps", nargs="+", default=["en-zh", "zh-en"])
    p.add_argument("--model_stem", required=True)
    p.add_argument("--hyps_model_stem", default=None, help="model_stem used for pcxmi_hyps (default: --model_stem)")
    p.add_argument("--full_context", default="full_context")
    p.add_argument("--no_context", default="no_context")
    p.add_argument("--targets", nargs="+", default=["ref", "hyp", "src"])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    root = Path(args.root_dir)
    summary = {}
    for ds in args.datasets:
        for lp in args.lps:
            key = f"{ds}.{lp}"
            summary[key] = {}
            for tgt in args.targets:
                fc = root / "pcxmi" / args.full_context / "mt" / key / args.model_stem / f"mean_log_probs_{tgt}.txt"
                nc = root / "pcxmi" / args.no_context / "mt" / key / args.model_stem / f"mean_log_probs_{tgt}.txt"
                if fc.exists() and nc.exists():
                    a, b = read_floats(fc), read_floats(nc)
                    d = a - b
                    summary[key][f"pcxmi_{tgt}"] = {
                        "mean": float(d.mean()),
                        "std": float(d.std()),
                        "frac_positive": float((d > 0).mean()),
                        "n": int(len(d)),
                    }
            hm = args.hyps_model_stem or args.model_stem
            for out_name in ["full_context_output", "no_context_output"]:
                fc = root / "pcxmi_hyps" / f"full_context_input_on_{out_name}" / "mt" / key / hm / "mean_log_probs.txt"
                nc = root / "pcxmi_hyps" / f"no_context_input_on_{out_name}" / "mt" / key / hm / "mean_log_probs.txt"
                if fc.exists() and nc.exists():
                    d = read_floats(fc) - read_floats(nc)
                    summary[key][f"pcxmi_on_{out_name}"] = {"mean": float(d.mean()), "std": float(d.std()), "n": int(len(d))}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
