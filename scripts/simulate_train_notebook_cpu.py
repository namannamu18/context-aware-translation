"""Executes the CODE CELLS of kaggle/train_eval_zh_kaggle.ipynb on CPU with the tiny stand-in models of scripts/smoke_models.py, inside a
scratch copy of Kaggle's folder layout. Only names, backend and sizes are replaced (the real values are Tower-7B, vLLM, 60 minutes, ...);
installs, the GPU query, the model download and the git clone are skipped. It tests the notebook's own code (settings, pre-flight, training,
merge, evaluation, checks, results table, packaging, failure handling), which the script tests alone do not.

  python scripts/simulate_train_notebook_cpu.py --smoke_dir smoke_models --sandbox /tmp/nbsim
(--smoke_dir contains tiny-chatml-lm/ and tiny-comet/checkpoints/model.ckpt)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke_dir", required=True)
    p.add_argument("--sandbox", required=True, help="scratch folder that plays the role of /kaggle")
    A = p.parse_args()
    S, K = str(Path(A.smoke_dir).resolve()), str(Path(A.sandbox).resolve())
    (Path(K) / "working").mkdir(parents=True, exist_ok=True)
    (Path(K) / "temp").mkdir(parents=True, exist_ok=True)
    link = Path(K) / "working" / "cat"
    if not link.exists():
        link.symlink_to(REPO)

    nb = json.load(open(REPO / "kaggle" / "train_eval_zh_kaggle.ipynb"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    skip = ("!pip", "!nvidia-smi", "snapshot_download", 'git", "clone')
    comet = f"{S}/tiny-comet/checkpoints/model.ckpt"
    repl = [("/kaggle/", K + "/"),
            ('assert "chat" not in MERGED_DIR.lower() and "chat" not in BASE_MODEL.lower()', "pass"),   # the tiny model is called tiny-chatml-lm
            ('BASE_MODEL   = "Unbabel/TowerInstruct-7B-v0.2"', f'BASE_MODEL   = "{S}/tiny-chatml-lm"'),
            ('"BACKEND": "vllm"', '"BACKEND": "hf"'), ("--gen_backend vllm", "--gen_backend hf"),
            ('"COMET_MODEL": "Unbabel/wmt22-comet-da"', f'"COMET_MODEL": "{comet}"'),
            ('"CD_EXTRA_ARGS": "--torch_dtype float16 --device_map auto"', '"CD_EXTRA_ARGS": "--torch_dtype float32"'),
            ('"SEP_TOKENS": "tower"', '"SEP_TOKENS": "chatml"'),
            ("TRAIN_MINUTES = 60", "TRAIN_MINUTES = 0.3"), ("TRAIN_GPUS    = 2", "TRAIN_GPUS    = 1"), ("EVAL_DOCS     = 12", "EVAL_DOCS     = 2"),
            ("N_CANDIDATES  = 6", "N_CANDIDATES  = 3"),
            ('TRANSLATE_EXTRA = ""', f'TRANSLATE_EXTRA = "--dtype float32 --device_map none --comet_model {comet}"')]
    ns = {"__name__": "__main__"}
    t0 = time.time()
    for i, src in enumerate(cells, 1):
        if any(m in src for m in skip):
            print(f"\n##### cell {i}: skipped (install / GPU / download / clone)")
            continue
        for a, b in repl:
            src = src.replace(a, b)
        print(f"\n##### cell {i}: {src.strip().splitlines()[0][:100]}", flush=True)
        exec(compile(src, f"cell{i}", "exec"), ns)
    ok = (Path(K) / "working" / "zh_lora" / "adapter_config.json").exists() and (Path(K) / "working" / "run_zh" / "results_zh" / "bmeld" / "test.en-zh.csv").exists()
    print(f"\nNOTEBOOK SIMULATION FINISHED in {(time.time() - t0) / 60:.1f} min; FAILED = {ns['FAILED']}; adapter and results table present: {ok}")
    sys.exit(0 if ok and not ns["FAILED"] else 1)


if __name__ == "__main__":
    main()
