"""Epsilon-sampling candidate generation for MBR (temperature 0.7, min_p 0.02,
100 candidates per segment).

Running the script without arguments reproduces the original run (vLLM,
TowerInstruct-7B-v0.2, bcontrast_test + wmt24_chat_test, all original language
pairs). en<->zh example (writes candidates/<prompt>/<model>/<dataset>.<lp>/):

  python scripts/generate_candidates.py --root_dir . --datasets bmeld_test \
      --lps en-zh zh-en --prompts no_context full_context
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_mt_utils import HFBackend, read_lines, vllm_engine_kwargs, write_lines  # noqa: E402

ORIGINAL_ROOT = "/mnt/data/jpombal/wmt24-chat-translation"
ORIGINAL_LPS = {
    "bcontrast_test": ["en-de", "de-en"],
    "wmt24_chat_test": [
        "en-de",
        "de-en",
        "en-fr",
        "fr-en",
        "en-ko",
        "ko-en",
        "en-nl",
        "nl-en",
        "en-pt-br",
        "pt-br-en",
    ],
    # additional English<->Chinese datasets (scripts/prepare_zh_data.py)
    "bmeld_test": ["en-zh", "zh-en"],
    "bmeld_dev": ["en-zh", "zh-en"],
    "synthetic_chat_test": ["en-zh", "zh-en"],
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default=ORIGINAL_ROOT)
    p.add_argument("--model", default="Unbabel/TowerInstruct-7B-v0.2")
    p.add_argument("--model_stem", default=None, help="Output folder name (default: basename of --model)")
    p.add_argument("--datasets", nargs="+", default=["bcontrast_test", "wmt24_chat_test"])
    p.add_argument("--lps", nargs="+", default=None, help="Default: all language pairs of each dataset")
    p.add_argument("--prompts", nargs="+", default=["no_context", "full_context"])
    p.add_argument("--n_candidates", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--min_p", type=float, default=0.02)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--limit", type=int, default=None, help="Only the first N segments (debugging)")
    return p.parse_args()


def main(args):
    model_stem = args.model_stem or Path(args.model).name
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams

        model = LLM(**vllm_engine_kwargs(model=args.model, seed=42, gpu_memory_utilization=0.9))
        # epsilon sampling
        s = SamplingParams(stop=None, max_tokens=args.max_tokens, temperature=args.temperature, min_p=args.min_p)

        def sample(prompts):
            return [o.outputs[0].text for o in model.generate(prompts, s, use_tqdm=True)]

    else:
        model = HFBackend(args.model, seed=42)

        def sample(prompts):
            return model.generate(prompts, temperature=args.temperature, min_p=args.min_p, max_tokens=args.max_tokens)

    for dataset in args.datasets:
        lps = args.lps or ORIGINAL_LPS[dataset]
        for p in args.prompts:
            for lp in lps:
                print(f"Generating candidates for {dataset} {p} {lp}")
                instructions = read_lines(
                    f"{args.root_dir}/instructions/{p}/mt/{dataset}.{lp}/instructions.txt",
                    unescape_newline=True,
                )
                if args.limit:
                    instructions = instructions[: args.limit]
                candidates = []
                for l in instructions:
                    candidates.extend([l] * args.n_candidates)
                generations = sample(candidates)
                write_lines(
                    f"{args.root_dir}/candidates/{p}/{model_stem}/{dataset}.{lp}/{args.n_candidates}_epsilon_candidates.txt",
                    generations,
                    escape_newline=True,
                )


if __name__ == "__main__":
    main(get_args())
