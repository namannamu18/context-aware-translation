"""Run tower-eval's own evaluation (`tower_eval.cli.run_evaluations`) for a
config in configs/ (e.g. configs/zh/full_context.yaml) on a machine without
vLLM. tower-eval imports vLLM at module load time even for evaluation, so a
stub module is registered when vLLM is not installed (generation is NOT
possible with the stub; use scripts/generate_translations.py for that).

Optional overrides make it possible to point a config at another output
root / model / metric subset without editing the YAML:

  python scripts/tower_eval_offline_evaluate.py --config configs/zh/full_context.yaml \
      --root_dir runs/smoke1 --metrics chrf bleu --model_type hf --model_name tiny-chatml-lm \
      --subtasks bmeld_test.en-zh bmeld_test.zh-en
"""

import argparse
import json
import os
import sys
import types
from pathlib import Path

import yaml


def ensure_vllm_importable():
    try:
        from vllm import LLM, SamplingParams  # noqa: F401
    except Exception:
        stub = types.ModuleType("vllm")

        class _Unavailable:
            def __init__(self, *a, **k):
                raise RuntimeError("vLLM is not installed; only evaluation is available")

        stub.LLM = _Unavailable
        stub.SamplingParams = _Unavailable
        stub.AsyncLLMEngine = _Unavailable
        stub.AsyncEngineArgs = _Unavailable
        sys.modules["vllm"] = stub
        for sub in ["vllm.lora", "vllm.lora.request", "vllm.sampling_params", "vllm.utils"]:
            m = types.ModuleType(sub)
            m.LoRARequest = _Unavailable
            m.random_uuid = lambda: "0"
            sys.modules[sub] = m
    # tower-eval also imports the SDKs of every API model provider at load time;
    # none of them is needed for evaluation, so stub the ones that are missing.
    import importlib.machinery
    import importlib.util

    class _StubFinder:
        roots = ("deepl", "vertexai", "anthropic", "cohere", "litellm", "google.generativeai", "google.cloud.aiplatform")

        def find_spec(self, name, path=None, target=None):
            if any(name == r or name.startswith(r + ".") for r in self.roots):
                return importlib.machinery.ModuleSpec(name, self, is_package=True)
            return None

        def create_module(self, spec):
            m = types.ModuleType(spec.name)
            m.__path__ = []
            m.__getattr__ = lambda attr: type(attr, (), {"__init__": lambda self, *a, **k: None})
            return m

        def exec_module(self, module):
            pass

    missing = [r for r in _StubFinder.roots if importlib.util.find_spec(r.split(".")[0]) is None]
    if missing:
        _StubFinder.roots = tuple(missing)
        sys.meta_path.append(_StubFinder())
    os.environ.setdefault("OPENAI_API_KEY", "unused")
    os.environ.setdefault("DEEPL_API_KEY", "unused")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--root_dir", default=None, help="Prefix for the relative paths in the config")
    p.add_argument("--metrics", nargs="+", default=None, help="Keep only these task metrics")
    p.add_argument("--subtasks", nargs="+", default=None)
    p.add_argument("--model_type", default=None)
    p.add_argument("--model_name", default=None)
    p.add_argument("--eval_output_root", default=None, help="Write evaluations under <this>/<condition> instead of the config's eval_output_dir")
    args = p.parse_args()

    ensure_vllm_importable()
    import contextlib

    with contextlib.redirect_stdout(sys.stderr):  # faiss prints its loader messages to stdout
        from tower_eval.cli import run_evaluations

    cfg = yaml.safe_load(open(args.config))
    if args.root_dir:
        for k in ["gen_data_dir", "eval_data_dir", "gen_output_dir", "eval_output_dir"]:
            cfg[k] = str(Path(args.root_dir) / cfg[k])
    if args.eval_output_root:
        cfg["eval_output_dir"] = str(Path(args.eval_output_root) / Path(cfg["eval_output_dir"]).name)
    for task in cfg["tasks"]:
        # same flattening as `tower-eval gen-eval`: subtask-level `eval_args:` (e.g. the zh / ko-mecab
        # BLEU tokenizer) become the subtask's evaluation arguments
        task["subtasks"] = {
            k: (v["eval_args"] if v and "eval_args" in v else v) for k, v in task["subtasks"].items()
        }
        if args.subtasks:
            task["subtasks"] = {k: v for k, v in task["subtasks"].items() if k in args.subtasks}
        if args.metrics:
            task["metrics"] = {k: v for k, v in task["metrics"].items() if k in args.metrics}
            for sub_args in task["subtasks"].values():
                if sub_args and "metrics" in sub_args:
                    sub_args["metrics"] = {k: v for k, v in sub_args["metrics"].items() if k in args.metrics}
    if args.model_type or args.model_name:
        cfg["models"] = [{"name": args.model_name or cfg["models"][0]["name"], "type": args.model_type or cfg["models"][0]["type"]}]
    run_evaluations(cfg)
    # print what tower-eval wrote (system-level scores only)
    summary = {}
    for model in cfg["models"]:
        for task in cfg["tasks"]:
            for subtask in task["subtasks"]:
                f = Path(cfg["eval_output_dir"]) / task["name"] / subtask / model["type"] / model["name"] / "evaluation.json"
                if f.exists():
                    res = json.load(open(f))
                    summary[f"{model['name']}/{task['name']}/{subtask}"] = {k: v for k, v in res.items() if not k.endswith("_segments")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
