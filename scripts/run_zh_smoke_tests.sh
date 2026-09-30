#!/bin/bash
# Three CPU smoke runs of the en<->zh pipeline on *different* test data, each followed by the
# consistency checks and by tower-eval's own evaluator on the configs/zh YAML.
#   run1: bmeld_test, first 8 conversations, COMET (smoke checkpoint) MBR
#   run2: bmeld_dev, conversations 21-28, chrF-utility MBR + COMET skipped if unavailable (fallback path)
#   run3: synthetic_chat_test (all 4 conversations), COMET (smoke checkpoint) MBR
# Requires smoke models (scripts/smoke_models.py). Outputs: $OUT/run{1,2,3}/ + logs.
# The smoke models are NOT research models: their scores are meaningless.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}
OUT=${OUT:?set OUT to an output folder}
LM=${LM:?path to smoke LM (scripts/smoke_models.py lm)}
COMET_CKPT=${COMET_CKPT:?path to smoke COMET model.ckpt (scripts/smoke_models.py comet)}
N=${N_CANDIDATES:-4}
MAXTOK=${MAX_TOKENS:-48}
mkdir -p "$OUT"
status=0

run() {  # name data split skip max utility comet_model
    local name=$1 data=$2 split=$3 skip=$4 max=$5 util=$6 comet=$7
    echo "### $name: $data $split skip=$skip max=$max utility=$util" | tee "$OUT/$name.log"
    local start=$(date +%s)
    PYTHON=$PY ROOT="$OUT/$name" DATA_NAME=$data SPLIT=$split SKIP_DOCS=$skip MAX_DOCS=$max BACKEND=hf SEP_TOKENS=chatml \
        MODEL="$LM" MODEL_NAME=tiny-chatml-lm N_CANDIDATES=$N MAX_TOKENS=$MAXTOK COMET_MODEL="$comet" \
        UTILITY=$util CONTEXT_SIZES="0 2" bash "$REPO/scripts/run_zh_pipeline.sh" >> "$OUT/$name.log" 2>&1
    local rc=$?
    echo "pipeline exit code: $rc ($(( $(date +%s) - start ))s)" | tee -a "$OUT/$name.log"
    $PY "$REPO/scripts/check_zh_pipeline_outputs.py" --root_dir "$OUT/$name" --data_name $data --split $split \
        --model "$LM" --model_name tiny-chatml-lm --n_candidates $N > "$OUT/$name.checks.txt" 2>/dev/null
    local rc2=$?
    tail -1 "$OUT/$name.checks.txt"
    $PY "$REPO/scripts/tower_eval_offline_evaluate.py" --config "$REPO/configs/zh/full_context.yaml" --root_dir "$OUT/$name" \
        --metrics chrf bleu --model_type hf --model_name tiny-chatml-lm --eval_output_root "$OUT/$name/evaluations_tower_eval" \
        --subtasks ${data}_${split}.en-zh ${data}_${split}.zh-en > "$OUT/$name.tower_eval.json" 2>/dev/null
    local rc3=$?
    echo "tower-eval evaluate exit code: $rc3"
    [ $rc -ne 0 ] || [ $rc2 -ne 0 ] || [ $rc3 -ne 0 ] && status=1
}

run run1 bmeld test 0 8 comet "$COMET_CKPT"
# run2: the real COMET id; where the HF Hub is unreachable this exercises the "no COMET" fallback
run run2 bmeld dev 20 8 chrf Unbabel/wmt22-comet-da
run run3 synthetic_chat test 0 "" comet "$COMET_CKPT"
exit $status
