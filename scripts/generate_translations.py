"""Greedy translation of instruction files, writing outputs in the same layout
as `tower-eval gen`:

  generations/<condition>/mt/<dataset>.<lp>/<backend>/<model_name>/generation.txt

This is what `tower-eval` does for `type: vllm` models (temperature 0,
max_tokens 1024, strip False). Use `tower-eval` with the configs in
`configs/zh/` when vLLM + GPU are available; use this script (with
`--backend hf`) to run the same step with HuggingFace transformers on CPU.

Example:
  python scripts/generate_translations.py --model Unbabel/TowerInstruct-7B-v0.2 \
      --model_name TowerInstruct-7B-v0.2 --conditions no_context full_context \
      --datasets bmeld_test --lps en-zh zh-en --backend vllm
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_mt_utils import HFBackend, read_lines, retry_degenerate, vllm_engine_kwargs, write_lines  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default=".")
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--model_name", required=True, help="Name of the output folder (as in tower-eval configs)")
    p.add_argument("--conditions", nargs="+", default=["no_context", "full_context"])
    p.add_argument("--datasets", nargs="+", default=["bmeld_test"])
    p.add_argument("--lps", nargs="+", default=["en-zh", "zh-en"])
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None, help="Only translate the first N segments (debugging)")
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                   help="1.0 = off (original setup). Applied to ALL outputs; prefer --loop_retry_penalty")
    p.add_argument("--loop_retry_penalty", type=float, default=0.0,
                   help="0 = off (original setup). If > 0, only outputs stuck in a repetition loop (e.g. '啊，啊，啊…') "
                        "are translated again with this repetition penalty (e.g. 1.1); all other outputs are unchanged")
    args = p.parse_args()
    root = Path(args.root_dir)

    if args.backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(**vllm_engine_kwargs(model=args.model))

        def gen(prompts, penalty=args.repetition_penalty):
            sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, repetition_penalty=penalty)
            return [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=True)]
    else:
        llm = HFBackend(args.model)

        def gen(prompts, penalty=args.repetition_penalty):
            return llm.generate(prompts, temperature=0.0, max_tokens=args.max_tokens, repetition_penalty=penalty)

    for cond in args.conditions:
        for ds in args.datasets:
            for lp in args.lps:
                inst_path = root / "instructions" / cond / "mt" / f"{ds}.{lp}" / "instructions.txt"
                prompts = read_lines(inst_path, unescape_newline=True)
                if args.limit:
                    prompts = prompts[: args.limit]
                print(f"Generating {cond} {ds}.{lp} ({len(prompts)} prompts)")
                outputs = gen(prompts)
                if args.loop_retry_penalty > 0:
                    outputs = retry_degenerate(prompts, outputs, lambda ps: gen(ps, args.loop_retry_penalty), f"{cond} {ds}.{lp}")
                out_dir = root / "generations" / cond / "mt" / f"{ds}.{lp}" / args.backend / args.model_name
                write_lines(out_dir / "generation.txt", outputs, escape_newline=True, verbose=False)
                with open(out_dir / "metadata.json", "w") as f:
                    json.dump(
                        {
                            "data_dir": str(inst_path.parent),
                            "model": args.model,
                            "model_name": args.model_name,
                            "backend": args.backend,
                            "temperature": 0.0,
                            "max_tokens": args.max_tokens,
                            "repetition_penalty": args.repetition_penalty,
                            "loop_retry_penalty": args.loop_retry_penalty,
                        },
                        f,
                        indent=4,
                    )


if __name__ == "__main__":
    main()
