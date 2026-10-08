#!/bin/bash
# CPU smoke test of the whole FINE-TUNED en<->zh workflow (what kaggle/train_eval_zh_kaggle.ipynb, judge_zh_kaggle.ipynb and
# translate_zh_kaggle.ipynb do on Kaggle), with the tiny test models of scripts/smoke_models.py:
#   LoRA fine-tuning -> merge -> base model greedy -> fine-tuned pipeline (_empty_sys prompts: greedy, candidates, MBR, contrastive
#   decoding, P-CXMI) -> consistency checks -> consolidated results table -> judge against a fake API server -> one-cell translator tests
# The smoke models are NOT research models: scores are meaningless, only the code paths are tested.
#
#   OUT=/tmp/ft_smoke LM=<tiny lm> COMET_CKPT=<tiny comet model.ckpt> bash scripts/run_zh_finetune_smoke_test.sh
#   NPROC=2 ...   also trains with two processes (gloo), the multi-GPU path of the Kaggle notebook
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}
OUT=${OUT:?set OUT to an output folder}
LM=${LM:?path to the smoke LM (scripts/smoke_models.py lm)}
COMET_CKPT=${COMET_CKPT:?path to the smoke COMET model.ckpt (scripts/smoke_models.py comet)}
NPROC=${NPROC:-1}
N=${N_CANDIDATES:-4}
MAXTOK=${MAX_TOKENS:-32}
mkdir -p "$OUT"
status=0
fail() { echo "FAILED: $*"; status=1; }

echo "### 1. LoRA fine-tuning (NPROC=$NPROC)"
if [ "$NPROC" -gt 1 ]; then LAUNCH=(torchrun --nproc_per_node "$NPROC" --master_port "${MASTER_PORT:-29571}"); else LAUNCH=("$PY"); fi
"${LAUNCH[@]}" "$REPO/scripts/finetune_lora.py" --base_model "$LM" --out_dir "$OUT/adapter" --max_examples 64 --max_steps 6 \
    --batch_size 2 --effective_batch 8 --eval_examples 16 --warmup_steps 2 > "$OUT/finetune.log" 2>&1 || fail "fine-tuning"
grep -E "train examples|trainable params|dev loss|FINETUNE_DONE" "$OUT/finetune.log" | cut -c1-220

echo "### 2. merge"
$PY "$REPO/scripts/merge_lora.py" --adapter "$OUT/adapter" --out_dir "$OUT/merged" --dtype float32 > "$OUT/merge.log" 2>&1 || fail "merge"
tail -1 "$OUT/merge.log"

ENVS=(PYTHON=$PY ROOT="$OUT/run" DATA_NAME=bmeld SPLIT=test MAX_DOCS=2 BACKEND=hf MAX_TOKENS=$MAXTOK COMET_MODEL="$COMET_CKPT" UTILITY=comet)

echo "### 3. base model, greedy"
env "${ENVS[@]}" MODEL="$LM" MODEL_NAME=tiny-base STAGES="data instructions greedy eval" SEP_TOKENS=chatml \
    bash "$REPO/scripts/run_zh_pipeline.sh" > "$OUT/pipeline_base.log" 2>&1 || fail "base pipeline"

echo "### 4. fine-tuned model: the paper's pipeline with the _empty_sys prompts"
env "${ENVS[@]}" MODEL="$OUT/merged" MODEL_NAME=tiny-ft PROMPT_SUFFIX=_empty_sys N_CANDIDATES=$N CONTEXT_SIZES="2" \
    CD_VARIANTS="c1_nc1 c5_nc1" CD_EXTRA_ARGS="--torch_dtype float32" SEP_TOKENS=chatml \
    STAGES="greedy eval candidates mbr cd pcxmi report" bash "$REPO/scripts/run_zh_pipeline.sh" > "$OUT/pipeline_ft.log" 2>&1 || fail "fine-tuned pipeline"

echo "### 5. consistency checks"
$PY "$REPO/scripts/check_zh_pipeline_outputs.py" --root_dir "$OUT/run" --data_name bmeld --split test --model "$OUT/merged" --model_name tiny-ft \
    --n_candidates $N --suffix _empty_sys --stages "greedy eval candidates mbr cd pcxmi" > "$OUT/checks.txt" 2>/dev/null || fail "consistency checks"
tail -1 "$OUT/checks.txt"

echo "### 6. consolidated results table"
$PY "$REPO/scripts/consolidate_zh_results.py" --root_dir "$OUT/run" --base_model_name tiny-base --ft_model_name tiny-ft --ft_suffix _empty_sys \
    --gen_backend hf --comet_model "$COMET_CKPT" > "$OUT/consolidate.log" 2>&1 || fail "consolidation"
grep -E "^wrote|skipping" "$OUT/consolidate.log"

echo "### 7. judge against a fake API server"
$PY "$REPO/scripts/test_judge_stub.py" "$OUT/run" > "$OUT/judge_test.log" 2>&1 || fail "judge test"
grep -E "TEST OK" "$OUT/judge_test.log"
$PY "$REPO/scripts/consolidate_zh_results.py" --root_dir "$OUT/run" --judge_only --judge_name stub > "$OUT/consolidate_judge.log" 2>&1 || fail "judge merge"
grep -E "^merged" "$OUT/consolidate_judge.log"

echo "### 8. one-cell translator"
$PY "$REPO/scripts/test_translate_chat_pipeline.py" --lm "$LM" --adapter "$OUT/adapter" --merged "$OUT/merged" --comet "$COMET_CKPT" \
    > "$OUT/translator_test.log" 2>&1 || fail "translator tests"
grep -E "^[0-5]\.|ALL TESTS" "$OUT/translator_test.log"

[ $status -eq 0 ] && echo "SMOKE TEST PASSED" || echo "SMOKE TEST FAILED (see the logs in $OUT)"
exit $status
