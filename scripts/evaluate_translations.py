"""Evaluate generations and write tower-eval compatible files:

  evaluations/<condition>/mt/<dataset>.<lp>/<backend>/<model_name>/evaluation.json
  -> {"chrf": .., "chrf_segments": [..], "bleu": .., "bleu_segments": [..], "comet": .., "comet_segments": [..]}

chrF/BLEU are computed with tower-eval's own metric classes when tower-eval
is importable, otherwise with the identical sacreBLEU calls. BLEU uses the
`zh` sacreBLEU tokenizer for Chinese targets (and `ko-mecab` for Korean, as in
the original configs). COMET (Unbabel/wmt22-comet-da by default) is computed
when the checkpoint can be loaded; otherwise it is skipped and the reason is
stored in metadata.json (use --require_comet to fail instead).

Example:
  python scripts/evaluate_translations.py --model_name TowerInstruct-7B-v0.2 \
     --conditions no_context full_context --datasets bmeld_test --lps en-zh zh-en
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_mt_utils import BLEU_TOKENIZER, load_jsonl, read_lines, split_lp  # noqa: E402


def _tower_eval_metrics():
    try:
        os.environ.setdefault("OPENAI_API_KEY", "unused")
        from tower_eval.metrics.bleu.metric import BLEU
        from tower_eval.metrics.chrf.metric import CHRF

        return CHRF, BLEU
    except Exception:
        return None, None


def chrf_bleu(hyps, refs, tgt_lang):
    CHRF, BLEU = _tower_eval_metrics()
    tok = BLEU_TOKENIZER.get(tgt_lang)
    if CHRF is not None:
        out = CHRF().evaluate(hyps, refs).format_result("chrf")
        out.update(BLEU().evaluate(hyps, refs, tokenize=tok).format_result("bleu"))
        return out, "tower-eval"
    from sacrebleu.metrics import BLEU as SBLEU
    from sacrebleu.metrics import CHRF as SCHRF

    chrf = SCHRF()
    bleu = SBLEU(tokenize=tok)
    bleu_seg = SBLEU(tokenize=tok, effective_order=True)
    return {
        "chrf": round(chrf.corpus_score(hyps, [refs]).score, 4),
        "chrf_segments": [chrf.sentence_score(h, [r]).score for h, r in zip(hyps, refs)],
        "bleu": round(bleu.corpus_score(hyps, [refs]).score, 4),
        "bleu_segments": [bleu_seg.sentence_score(h, [r]).score for h, r in zip(hyps, refs)],
    }, "sacrebleu"


_COMET = {}


def load_comet(name):
    if name not in _COMET:
        from comet import download_model, load_from_checkpoint

        path = name if name.endswith(".ckpt") else download_model(name)
        _COMET[name] = load_from_checkpoint(path)
    return _COMET[name]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default=".")
    p.add_argument("--model_name", required=True)
    p.add_argument("--backend", default="vllm", help="Generation sub-folder (vllm for tower-eval / vllm, hf for the HF backend)")
    p.add_argument("--conditions", nargs="+", default=["no_context", "full_context"])
    p.add_argument("--datasets", nargs="+", default=["bmeld_test"])
    p.add_argument("--lps", nargs="+", default=["en-zh", "zh-en"])
    p.add_argument("--comet_model", default="Unbabel/wmt22-comet-da", help="HF id or local .ckpt; 'none' to disable")
    p.add_argument("--require_comet", action="store_true")
    p.add_argument("--gpus", type=int, default=None)
    args = p.parse_args()
    root = Path(args.root_dir)

    for cond in args.conditions:
        for ds in args.datasets:
            for lp in args.lps:
                gen_dir = root / "generations" / cond / "mt" / f"{ds}.{lp}" / args.backend / args.model_name
                hyps = read_lines(gen_dir / "generation.txt", unescape_newline=True)
                data = load_jsonl(root / "raw_data" / "mt" / f"{ds}.{lp}" / "test.jsonl")[: len(hyps)]
                srcs, refs = [d["src"] for d in data], [d["ref"] for d in data]
                assert len(hyps) == len(refs), (len(hyps), len(refs))
                _, tgt = split_lp(lp)
                result, impl = chrf_bleu(hyps, refs, tgt)
                meta = {"lexical_metrics_impl": impl, "bleu_tokenizer": BLEU_TOKENIZER.get(tgt, "13a")}
                if args.comet_model.lower() != "none":
                    try:
                        m = load_comet(args.comet_model)
                        gpus = args.gpus if args.gpus is not None else (1 if _cuda() else 0)
                        o = m.predict(
                            [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)],
                            batch_size=16,
                            gpus=gpus,
                            progress_bar=False,
                        )
                        result["comet"] = round(float(o.system_score), 4)
                        result["comet_segments"] = [float(x) for x in o.scores]
                        meta["comet_model"] = args.comet_model
                    except Exception as e:  # e.g. no network access to the HF hub
                        if args.require_comet:
                            raise
                        meta["comet_skipped"] = f"{type(e).__name__}: {str(e)[:300]}"
                out_dir = root / "evaluations" / cond / "mt" / f"{ds}.{lp}" / args.backend / args.model_name
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "evaluation.json", "w") as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                with open(out_dir / "metadata.json", "w") as f:
                    json.dump(meta, f, indent=4)
                summary = {k: v for k, v in result.items() if not k.endswith("_segments")}
                print(f"{cond} {ds}.{lp}: {summary}" + (" (COMET skipped)" if "comet_skipped" in meta else ""))


def _cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
