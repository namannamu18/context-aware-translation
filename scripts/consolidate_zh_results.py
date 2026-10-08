"""Collect every system of an English<->Chinese run into ONE csv, like `paper_results/{dev,test}.<lp>.csv` of the paper:
one row per message (both directions, in conversation order), one column per system, a `<system>-comet` column with
its COMET-22 segment score, and (after scripts/run_context_llm.py) a `<system>-mqm` column with the judge's score.
Also writes a summary table (chrF / BLEU / COMET per direction and system).

Systems (names follow the paper's table; "ft" = our LoRA fine-tuned chat model, see scripts/finetune_lora.py):
  greedy-7b-wo-context, greedy-7b-w-context          base TowerInstruct-7B, greedy, without / with context
  greedy-7b-ft-wo-context, greedy-7b-ft-w-context    fine-tuned model, greedy
  mbr-wo-context                                      fine-tuned model, candidates without context, COMET MBR
  mbr                                                 fine-tuned model, candidates with context, COMET MBR
  mbr-source   (the paper's PRIMARY system)           ... context-aware COMET MBR, context = previous source messages
  mbr-hyp                                             ... context-aware COMET MBR, context = best earlier translations
  cd-<variant>                                        fine-tuned model, contrastive decoding (c1_nc1, c5_nc1, ...)

  python scripts/consolidate_zh_results.py --root_dir run --base_model_name TowerInstruct-7B-v0.2 \\
      --ft_model_name TowerInstruct-7B-w-chat-zh --ft_suffix _empty_sys
  python scripts/consolidate_zh_results.py --root_dir run --judge_only --judge_name gemini     # add the judge's scores later
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_mt_utils import read_lines  # noqa: E402

PRIMARY = "mbr-source"


def interleave(per_lp, lps):
    """Per-direction files -> one list in the row order of the bilingual csv (same logic as get_translations in the MBR script)."""
    pos = {k: 0 for k in per_lp}
    out = []
    for lp in lps:
        out.append(per_lp[lp][pos[lp]])
        pos[lp] += 1
    assert all(pos[k] == len(per_lp[k]) for k in per_lp), "direction files do not match the data"
    return out


def greedy(root, ds, cond, model, backend, lps):
    per = {}
    for lp in ("en-zh", "zh-en"):
        f = root / "generations" / cond / "mt" / f"{ds}.{lp}" / backend / model / "generation.txt"
        if not f.exists():
            return None
        per[lp] = read_lines(f, unescape_newline=True)
    return interleave(per, lps)


def lexical(hyps, refs, tgt):
    hyps = ["" if (isinstance(h, float) and np.isnan(h)) else str(h) for h in hyps]
    return {"chrf": round(CHRF().corpus_score(hyps, [list(refs)]).score, 2),
            "bleu": round(BLEU(tokenize="zh" if tgt == "zh" else "13a").corpus_score(hyps, [list(refs)]).score, 2)}


def merge_judge(df, out_dir, data_name, split, judge_name):
    """Add `<system>-mqm` columns from the files written by scripts/run_context_llm.py (rows are matched by doc_id + position in the conversation)."""
    seg = df.groupby("doc_id").cumcount()
    key = df["doc_id"].astype(str) + "#" + seg.astype(str)
    for f in sorted(glob.glob(str(out_dir / data_name / f"{split}.en-zh-*.gemba-{judge_name}*.csv"))):
        m = re.search(rf"{re.escape(split)}\.en-zh-(.+?)\.gemba-", Path(f).name)
        if not m:
            continue
        system = m.group(1)
        j = pd.read_csv(f, keep_default_na=False, na_values=[])
        score_col = [c for c in j.columns if c.endswith("-score")]
        if not score_col:
            continue
        jk = j["doc_id"].astype(str) + "#" + j["segment_id"].astype(str)
        df[f"{system}-mqm"] = key.map(dict(zip(jk, pd.to_numeric(j[score_col[0]], errors="coerce"))))
        print(f"merged judge scores for {system}: {df[f'{system}-mqm'].notna().sum()} rows")
    return df


def summarize(df, systems):
    table = {}
    for name in systems:
        table[name] = {}
        for lp, g in df.groupby("lp"):
            tgt = lp.split("-")[1]
            entry = lexical(g[name], g["reference"], tgt)
            if f"{name}-comet" in g:
                c = pd.to_numeric(g[f"{name}-comet"], errors="coerce")
                if c.notna().any():
                    entry["comet"] = round(float(c.mean()), 4)
            if f"{name}-mqm" in g:
                q = pd.to_numeric(g[f"{name}-mqm"], errors="coerce")
                if q.notna().any():
                    entry["mqm"] = round(float(q.mean()), 3)
            entry["n"] = len(g)
            table[name][lp] = entry
    return table


def to_markdown(table):
    lps = sorted({lp for v in table.values() for lp in v})
    lines = ["| system | " + " | ".join(f"{lp} chrF | {lp} BLEU | {lp} COMET | {lp} MQM" for lp in lps) + " |",
             "|---|" + "---|" * (4 * len(lps))]
    for name, v in table.items():
        cells = []
        for lp in lps:
            e = v.get(lp, {})
            cells += [str(e.get("chrf", "")), str(e.get("bleu", "")), str(e.get("comet", "")), str(e.get("mqm", ""))]
        lines.append(f"| {name}{' (primary)' if name == PRIMARY else ''} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", required=True)
    p.add_argument("--data_name", default="bmeld")
    p.add_argument("--split", default="test")
    p.add_argument("--base_model_name", default=None, help="model name of the base-model run (greedy baselines)")
    p.add_argument("--base_suffix", default="")
    p.add_argument("--ft_model_name", default=None, help="model name of the fine-tuned run (greedy, MBR, contrastive decoding)")
    p.add_argument("--ft_suffix", default="_empty_sys")
    p.add_argument("--gen_backend", default="vllm")
    p.add_argument("--context_size", type=int, default=2, help="window of the context-aware MBR variants")
    p.add_argument("--comet_model", default="Unbabel/wmt22-comet-da", help="HF id, local .ckpt, or 'none'")
    p.add_argument("--out_dir", default=None, help="default: <root_dir>/results_zh")
    p.add_argument("--judge_only", action="store_true", help="only merge the judge's scores into the existing csv")
    p.add_argument("--judge_name", default="gemini", help="model_name used with run_context_llm.py")
    args = p.parse_args()

    root = Path(args.root_dir)
    ds = f"{args.data_name}_{args.split}"
    out_dir = Path(args.out_dir) if args.out_dir else root / "results_zh"
    out_csv = out_dir / args.data_name / f"{args.split}.en-zh.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.judge_only:
        df = pd.read_csv(out_csv, keep_default_na=False, na_values=[])
        systems = [c for c in df.columns if f"{c}-comet" in df.columns]
        df = merge_judge(df, out_dir, args.data_name, args.split, args.judge_name)
    else:
        base = pd.read_csv(root / "paper_results_zh" / args.data_name / f"{args.split}.en-zh.csv", keep_default_na=False, na_values=["NaN"])
        df = base.copy()
        lps = df["lp"].tolist()
        columns = {}
        if args.base_model_name:
            for name, cond in (("greedy-7b-wo-context", "no_context"), ("greedy-7b-w-context", "full_context")):
                columns[name] = greedy(root, ds, cond + args.base_suffix, args.base_model_name, args.gen_backend, lps)
        if args.ft_model_name:
            sfx = args.ft_suffix
            for name, cond in (("greedy-7b-ft-wo-context", "no_context"), ("greedy-7b-ft-w-context", "full_context")):
                columns[name] = greedy(root, ds, cond + sfx, args.ft_model_name, args.gen_backend, lps)
            w = args.context_size
            mbr = {"mbr-wo-context": ("no_context", "comet_eps"), "mbr": ("full_context", "comet_eps"),
                   "mbr-source": ("full_context", f"comet_eps_context_source_w{w}"),
                   "mbr-hyp": ("full_context", f"comet_eps_context_comet-best_w{w}")}
            for name, (cond, fname) in mbr.items():
                f = root / "mbr_outputs" / f"mbr_outputs_{args.split}" / (cond + sfx) / args.ft_model_name / f"{ds}.en-zh" / f"{fname}.csv"
                if not f.exists():
                    columns[name] = None
                    continue
                m = pd.read_csv(f, keep_default_na=False, na_values=["NaN"])
                assert m["source"].tolist() == df["source"].tolist(), f"{f} is not aligned with the data"
                columns[name] = [("" if pd.isna(x) else str(x)) for x in m["output-select"]]
            for f in sorted(glob.glob(str(root / "contrast_decode" / f"{ds}.en-zh" / "*.out.txt"))):
                name = "cd-" + Path(f).name.replace(".out.txt", "")
                lines = open(f, encoding="utf-8").read().split("\n")[:-1]
                columns[name] = lines if len(lines) == len(df) else None   # sampled variants (several outputs per message) are skipped
        systems = []
        for name, col in columns.items():
            if col is None:
                print(f"(skipping {name}: outputs not found)")
                continue
            df[name] = col
            systems.append(name)

        if args.comet_model.lower() != "none" and systems:
            from comet import download_model, load_from_checkpoint
            import torch

            comet = load_from_checkpoint(args.comet_model if args.comet_model.endswith(".ckpt") else download_model(args.comet_model))
            for name in systems:
                o = comet.predict([{"src": s, "mt": h, "ref": r} for s, h, r in zip(df["source"], df[name], df["reference"])],
                                  batch_size=16, gpus=1 if torch.cuda.is_available() else 0, progress_bar=False)
                df[f"{name}-comet"] = [float(x) for x in o.scores]
        df = merge_judge(df, out_dir, args.data_name, args.split, args.judge_name)

    df.to_csv(out_csv, index=False)
    table = summarize(df, systems)
    with open(out_dir / args.data_name / f"{args.split}.en-zh.summary.json", "w") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    md = to_markdown(table)
    (out_dir / args.data_name / f"{args.split}.en-zh.summary.md").write_text(md + "\n", encoding="utf-8")
    print(f"wrote {out_csv} ({len(df)} rows, {len(systems)} systems)\n")
    print(md)


if __name__ == "__main__":
    main()
