"""English<->Chinese version of the fine-tuning data preparation
(wo_context.py / w_context.py / add_chat_data.py).

Builds chat-MT fine-tuning examples in the TowerBlocks "conversations" format
from the BMELD training split (scripts/prepare_zh_data.py), with or without
conversational context. Targets are either the references (as in
w_context.py / wo_context.py) or MBR-selected translations from a
run_context_comet_mbr.py output csv (as in add_chat_data.py: "ctx comet mbr
distill"). The result is written to a local jsonl; mixing with TowerBlocks and
uploading to the Hub are left to the user (the original scripts push to
private Unbabel repos).

  python scripts/training_tower_chat/make_zh_chat_mt_data.py --context full --out chat_mt_zh_w_context.jsonl
  python scripts/training_tower_chat/make_zh_chat_mt_data.py --context none --out chat_mt_zh_no_context.jsonl
  python scripts/training_tower_chat/make_zh_chat_mt_data.py --context full --mbr_csv mbr_outputs/.../comet_eps_context_source_w2.csv --out ...
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat_mt_utils import build_prompts, load_jsonl  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default=".")
    p.add_argument("--dataset", default="bmeld_train")
    p.add_argument("--context", choices=["none", "full"], default="full")
    p.add_argument("--mbr_csv", default=None, help="Use the 'output-select' column of this MBR csv as targets")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    recs = load_jsonl(Path(args.root_dir) / "raw_data" / "mt" / f"{args.dataset}.zh" / "test.jsonl")
    # raw user messages (no chat template), as in add_chat_data.py; the trainer applies the template
    prompts = build_prompts(recs, "no_template", use_context=args.context == "full")
    targets = [r["ref"] for r in recs]
    name = f"{args.dataset}_{'w' if args.context == 'full' else 'wo'}_context"
    if args.mbr_csv:
        mbr = pd.read_csv(args.mbr_csv, keep_default_na=False, na_values=["NaN"])
        assert len(mbr) == len(recs) and (mbr["source"].tolist() == [r["src"] for r in recs])
        targets = mbr["output-select"].tolist()
        name += "_ctx_comet_mbr_distill"
    df = pd.DataFrame(
        {
            "conversations": [[{"from": "human", "value": p_}, {"from": "gpt", "value": t}] for p_, t in zip(prompts, targets)],
            "lang": [f"{r['source_language']}-{r['target_language']}" for r in recs],
            "split": "train",
            "dataset": name,
            "task": "chat_translation",
        }
    )
    df.to_json(args.out, orient="records", lines=True, force_ascii=False)
    print(f"wrote {len(df)} examples to {args.out}")


if __name__ == "__main__":
    main()
