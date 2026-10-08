"""Consistency checks for one run of scripts/run_zh_pipeline.sh (en-zh and zh-en).

Checks (each printed as PASS/FAIL):
  * instructions: one prompt per segment, prompt names the right languages, the
    full-context prompt contains the previous utterances of the conversation;
  * generations / candidates / contrastive outputs have the expected number of lines;
  * MBR: every selected translation is one of that segment's candidates and rows
    are aligned with the source data;
  * P-CXMI: for every segment, #ref log-probs == #tokens(reference) + 1 (EOS),
    i.e. the prompt/answer gating found the right position, for both context settings;
    #src log-probs == #tokens(source);
  * evaluation.json files contain chrF/BLEU with one segment score per segment;
  * language sanity (informative only, not a failure): share of CJK characters in
    en->zh vs zh->en outputs.

Exit code 1 if any check fails.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_mt_utils import CODE_LANG_DICT, load_jsonl, read_lines, split_lp  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def cjk_ratio(texts):
    chars = "".join(texts)
    if not chars:
        return 0.0
    return sum("一" <= c <= "鿿" for c in chars) / max(1, sum(not c.isspace() for c in chars))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", required=True)
    p.add_argument("--data_name", default="bmeld")
    p.add_argument("--split", default="test")
    p.add_argument("--model", required=True, help="Model path (to load the tokenizer)")
    p.add_argument("--model_name", required=True)
    p.add_argument("--gen_backend", default="hf")
    p.add_argument("--n_candidates", type=int, required=True)
    p.add_argument("--suffix", default="", help='prompt suffix of the run, e.g. "_empty_sys" (PROMPT_SUFFIX of run_zh_pipeline.sh)')
    p.add_argument("--stages", default="all", help="stages the run executed (STAGES of run_zh_pipeline.sh); only those are checked")
    args = p.parse_args()
    want = lambda stage: args.stages == "all" or stage in args.stages.split()  # noqa: E731
    NC, FC = "no_context" + args.suffix, "full_context" + args.suffix
    CONDS = [NC, FC]
    root = Path(args.root_dir)
    ds = f"{args.data_name}_{args.split}"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    lang_share = {}

    for lp in ["en-zh", "zh-en"]:
        s, t = split_lp(lp)
        data = load_jsonl(root / "raw_data" / "mt" / f"{ds}.{lp}" / "test.jsonl")
        n = len(data)
        check(f"{lp}: raw data non-empty and directions correct", n > 0 and all(d["source_language"] == s and d["target_language"] == t for d in data), f"n={n}")
        # instructions
        for cond in CONDS:
            inst = read_lines(root / "instructions" / cond / "mt" / f"{ds}.{lp}" / "instructions.txt", unescape_newline=True)
            ok = len(inst) == n and all(f"\n{CODE_LANG_DICT[s]}: {d['src']}\n{CODE_LANG_DICT[t]}: " in i for i, d in zip(inst, data))
            check(f"{lp}: {cond} instructions aligned with sources", ok, f"{len(inst)} prompts")
        # full-context prompt contains previous utterances of the same conversation
        bil = load_jsonl(root / "raw_data" / "mt" / f"{ds}.zh" / "test.jsonl")
        inst_fc = read_lines(root / "instructions" / FC / "mt" / f"{ds}.{lp}" / "instructions.txt", unescape_newline=True)
        ok, k, prev_doc, history = True, 0, None, []
        for r in bil:
            if r["doc_id"] != prev_doc:
                history, prev_doc = [], r["doc_id"]
            if r["source_language"] == s:
                if history:
                    ok &= inst_fc[k].split("\n\nTranslate the")[0].endswith("\n".join(history[-1:]))
                    ok &= all(h in inst_fc[k] for h in history)
                else:
                    ok &= "Context:" not in inst_fc[k]
                k += 1
            history.append(r["src"])
        check(f"{lp}: full_context prompts contain the conversation history (both speakers)", ok)
        # generations, evaluation, candidates
        for cond in CONDS:
            if want("greedy"):
                g = root / "generations" / cond / "mt" / f"{ds}.{lp}" / args.gen_backend / args.model_name / "generation.txt"
                hyps = read_lines(g, unescape_newline=True)
                check(f"{lp}: {cond} greedy generations", len(hyps) == n, f"{len(hyps)} lines")
                lang_share[f"{cond} {lp} hyps CJK share"] = round(cjk_ratio(hyps), 3)
            if want("eval"):
                ev = json.load(open(root / "evaluations" / cond / "mt" / f"{ds}.{lp}" / args.gen_backend / args.model_name / "evaluation.json"))
                check(f"{lp}: {cond} evaluation.json", all(len(ev.get(f"{m}_segments", [])) == n for m in ["chrf", "bleu"]), ", ".join(f"{k}={v}" for k, v in ev.items() if not k.endswith("_segments")))
            if want("candidates"):
                cands = read_lines(root / "candidates" / cond / args.model_name / f"{ds}.{lp}" / f"{args.n_candidates}_epsilon_candidates.txt")
                check(f"{lp}: {cond} candidates", len(cands) == n * args.n_candidates, f"{len(cands)} = {n} x {args.n_candidates}")
        lang_share[f"{lp} refs CJK share"] = round(cjk_ratio([d["ref"] for d in data]), 3)
        # PCXMI gating
        for cond in (CONDS if want("pcxmi") else []):
            pdir = root / "pcxmi" / cond / "mt" / f"{ds}.{lp}" / args.model_name
            ref_lps = load_jsonl(pdir / "log_probs_ref.jsonl")
            exp = [len(tok.encode(d["ref"], add_special_tokens=False)) + 1 for d in data]
            got = [len(x["log_probs"]) for x in ref_lps]
            check(f"{lp}: P-CXMI ref gating ({cond})", got == exp, f"{sum(a == b for a, b in zip(got, exp))}/{n} segments match")
            src_lps = load_jsonl(pdir / "log_probs_src.jsonl")
            exp_s = [len(tok.encode(d["src"], add_special_tokens=False)) for d in data]
            got_s = [len(x["log_probs"]) for x in src_lps]
            check(f"{lp}: P-CXMI src gating ({cond})", got_s == exp_s, f"{sum(a == b for a, b in zip(got_s, exp_s))}/{n}")
            hyp_lps = load_jsonl(pdir / "log_probs_hyp.jsonl")
            check(f"{lp}: P-CXMI hyp log-probs ({cond})", len(hyp_lps) == n and all(len(x["log_probs"]) > 0 for x in hyp_lps))
        for combo in (["full_context_input_on_full_context_output", "full_context_input_on_no_context_output", "no_context_input_on_full_context_output", "no_context_input_on_no_context_output"] if want("pcxmi") else []):
            f = root / "pcxmi_hyps" / combo / "mt" / f"{ds}.{lp}" / args.model_name / "mean_log_probs.txt"
            check(f"{lp}: pcxmi_hyps {combo}", f.exists() and len(read_lines(f)) == n)

    # MBR
    pr = pd.read_csv(root / "paper_results_zh" / args.data_name / f"{args.split}.en-zh.csv", keep_default_na=False, na_values=["NaN"])
    for csv in (sorted(glob.glob(str(root / "mbr_outputs" / "**" / "*.csv"), recursive=True)) if want("mbr") else []):
        df = pd.read_csv(csv, keep_default_na=False, na_values=["NaN"])
        cond = Path(csv).parts[-4]
        cand = {}
        for lp in ["en-zh", "zh-en"]:
            c = read_lines(root / "candidates" / cond / args.model_name / f"{ds}.{lp}" / f"{args.n_candidates}_epsilon_candidates.txt", unescape_newline=True)
            cand[lp] = np.array(c, dtype=object).reshape(-1, args.n_candidates)
        idx = {"en-zh": 0, "zh-en": 0}
        ok = len(df) == len(pr) and df["source"].tolist() == pr["source"].tolist()
        for _, r in df.iterrows():
            lp = r["lp"]
            sel = "" if pd.isna(r["output-select"]) else str(r["output-select"])
            # run_context_comet_mbr.read_file() strips surrounding whitespace (original behaviour)
            ok &= sel.strip() in [str(x).strip() for x in cand[lp][idx[lp]]]
            idx[lp] += 1
        check(f"MBR {Path(csv).relative_to(root / 'mbr_outputs')}: selections are candidates of the right segment", ok)

    # contrastive decoding
    import re

    for out in (sorted(glob.glob(str(root / "contrast_decode" / f"{ds}.en-zh" / "*.out.txt"))) if want("cd") else []):
        lines = open(out).read().split("\n")[:-1]
        m = re.search(r"_n(\d+)\.out\.txt$", out)       # sampled variants: n outputs per segment
        mult = int(m.group(1)) if m else 1
        check(f"contrastive {Path(out).name}", len(lines) == len(pr) * mult, f"{len(lines)} lines")

    print("\nlanguage sanity (informative):", json.dumps(lang_share))
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
