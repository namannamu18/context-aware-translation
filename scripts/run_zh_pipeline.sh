#!/bin/bash
# End-to-end English<->Chinese run of the (non-paid) pipeline:
#   data subset -> instructions (no/full context) -> greedy decoding -> evaluation
#   -> epsilon-sampling candidates -> MBR (no-context COMET, context-COMET, comet-best context)
#   -> contrastive (context vs. no-context) decoding -> P-CXMI (ref/hyp/src + cross-condition) -> report
#
# All settings come from environment variables (defaults = real setup on a GPU box):
#   ROOT=.                       where outputs are written (a copy of the data is made when MAX_DOCS is set)
#   DATA_NAME=bmeld SPLIT=test   dataset (bmeld_{train,dev,test} or synthetic_chat_test)
#   MODEL=Unbabel/TowerInstruct-7B-v0.2  MODEL_NAME=TowerInstruct-7B-v0.2
#   BACKEND=vllm                 vllm | hf (HuggingFace transformers, CPU friendly)
#   SEP_TOKENS=tower             tower (Tower token ids, as in the paper) | chatml (derived from tokenizer)
#   N_CANDIDATES=100             epsilon-sampling candidates per segment (paper: 100)
#   COMET_MODEL=Unbabel/wmt22-comet-da   HF id or local .ckpt
#   UTILITY=comet                comet (paper) | chrf (fallback when no COMET checkpoint is available)
#   MAX_DOCS=                    if set, only the first MAX_DOCS conversations (after SKIP_DOCS) are used
#   SKIP_DOCS=0
#   CONTEXT_SIZES="0 2"          context window(s) for context-aware COMET MBR (paper: 0 2 6 10 15)
#   MAX_TOKENS=1024
#   REPETITION_PENALTY=1.0       greedy decoding, applied to all outputs; 1.0 = off (paper setup)
#   LOOP_RETRY_PENALTY=0         0 = off (paper setup); e.g. 1.1 = only outputs stuck in a repetition loop ("啊，啊，啊…")
#                                are translated again with that penalty, all other outputs stay identical
#   VLLM_ENGINE_ARGS=            JSON passed to vllm.LLM, e.g. '{"dtype":"half","tensor_parallel_size":2}' on 2x T4
#   CD_EXTRA_ARGS=               extra args for contrastive decoding, e.g. "--torch_dtype float16 --device_map auto"
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}
ROOT=${ROOT:-.}
DATA_NAME=${DATA_NAME:-bmeld}
SPLIT=${SPLIT:-test}
MODEL=${MODEL:-Unbabel/TowerInstruct-7B-v0.2}
MODEL_NAME=${MODEL_NAME:-TowerInstruct-7B-v0.2}
BACKEND=${BACKEND:-vllm}
SEP_TOKENS=${SEP_TOKENS:-tower}
N_CANDIDATES=${N_CANDIDATES:-100}
COMET_MODEL=${COMET_MODEL:-Unbabel/wmt22-comet-da}
UTILITY=${UTILITY:-comet}
MAX_DOCS=${MAX_DOCS:-}
SKIP_DOCS=${SKIP_DOCS:-0}
CONTEXT_SIZES=${CONTEXT_SIZES:-"0 2"}
MAX_TOKENS=${MAX_TOKENS:-1024}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
LOOP_RETRY_PENALTY=${LOOP_RETRY_PENALTY:-0}
DS=${DATA_NAME}_${SPLIT}
LPS="en-zh zh-en"

mkdir -p "$ROOT"
ROOT=$(cd "$ROOT" && pwd)
echo "### run_zh_pipeline: ROOT=$ROOT DS=$DS MODEL=$MODEL BACKEND=$BACKEND N=$N_CANDIDATES UTILITY=$UTILITY"

step() { echo; echo "=================== $* ==================="; }

# 1. data -----------------------------------------------------------------------------------
step "1. data"
if [ -n "$MAX_DOCS" ]; then
    $PY "$REPO/scripts/subset_zh_dataset.py" --src_root "$REPO" --dst_root "$ROOT" --data_name "$DATA_NAME" \
        --split "$SPLIT" --max_docs "$MAX_DOCS" --skip_docs "$SKIP_DOCS"
elif [ "$ROOT" != "$REPO" ]; then
    $PY "$REPO/scripts/subset_zh_dataset.py" --src_root "$REPO" --dst_root "$ROOT" --data_name "$DATA_NAME" --split "$SPLIT"
fi

# 2. instructions ---------------------------------------------------------------------------
step "2. instructions"
$PY "$REPO/scripts/make_instructions.py" --root_dir "$ROOT" --datasets "$DS" --pairs en-zh \
    --conditions no_context full_context no_context_empty_sys full_context_empty_sys full_context_6_turns

# 3. greedy decoding (tower-eval gen equivalent) + 4. evaluation ------------------------------
step "3. greedy decoding (no_context / full_context)"
$PY "$REPO/scripts/generate_translations.py" --root_dir "$ROOT" --model "$MODEL" --model_name "$MODEL_NAME" \
    --conditions no_context full_context --datasets "$DS" --lps $LPS --backend "$BACKEND" --max_tokens "$MAX_TOKENS" \
    --repetition_penalty "$REPETITION_PENALTY" --loop_retry_penalty "$LOOP_RETRY_PENALTY"
step "4. evaluation"
$PY "$REPO/scripts/evaluate_translations.py" --root_dir "$ROOT" --model_name "$MODEL_NAME" --backend "$BACKEND" \
    --conditions no_context full_context --datasets "$DS" --lps $LPS --comet_model "$COMET_MODEL"

# 5. candidates (epsilon sampling: temperature 0.7, min_p 0.02) ------------------------------
step "5. epsilon-sampling candidates"
$PY "$REPO/scripts/generate_candidates.py" --root_dir "$ROOT" --model "$MODEL" --model_stem "$MODEL_NAME" \
    --datasets "$DS" --lps $LPS --prompts no_context full_context --n_candidates "$N_CANDIDATES" \
    --backend "$BACKEND" --max_tokens "$MAX_TOKENS"

# 6. MBR / QAD --------------------------------------------------------------------------------
step "6. MBR decoding"
cd "$ROOT"  # run_context_comet_mbr.py reads candidates/ and generations/ relative to the cwd
for context in no_context full_context; do
    OUT_DIR=mbr_outputs/mbr_outputs_${SPLIT}/${context}/${MODEL_NAME}/${DS}.en-zh
    mkdir -p "$OUT_DIR"
    MBR_ARGS=(--lang_pair en-zh --split "$SPLIT" --data_name "$DATA_NAME" --data_dir "$ROOT/paper_results_zh"
              --inf_context "$context" --model_name "$MODEL_NAME" --num_candidates "$N_CANDIDATES"
              --gen_backend "$BACKEND" --comet_model "$COMET_MODEL" --utility "$UTILITY" --batch_size 16)
    # (a) standard (source-only) COMET MBR
    $PY "$REPO/scripts/run_context_comet_mbr.py" "${MBR_ARGS[@]}" --save_output "$OUT_DIR/comet_eps"
    for w in $CONTEXT_SIZES; do
        # (b) context-aware COMET MBR, context = previous source utterances
        $PY "$REPO/scripts/run_context_comet_mbr.py" "${MBR_ARGS[@]}" --use_context --context_size "$w" \
            --context_source source --context_mt source --save_output "$OUT_DIR/comet_eps_context_source_w${w}"
        # (c) context-aware COMET MBR, context = COMET-MBR-selected translations of previous utterances
        $PY "$REPO/scripts/run_context_comet_mbr.py" "${MBR_ARGS[@]}" --use_context --context_size "$w" \
            --context_source source --context_mt comet-best --save_output "$OUT_DIR/comet_eps_context_comet-best_w${w}"
    done
done
cd - > /dev/null

# 7. contrastive decoding (context vs. no-context prompts) ------------------------------------
step "7. contrastive decoding"
CD_DIR=$ROOT/contrast_decode/${DS}.en-zh
mkdir -p "$CD_DIR"
EVAL_FLAG=()
[ "$UTILITY" = "comet" ] && EVAL_FLAG=(--eval --comet_model "$COMET_MODEL")
CD_ARGS=(--lang_pair en-zh --split "$SPLIT" --data_name "$DATA_NAME" --data_dir "$ROOT/paper_results_zh/$DATA_NAME"
         --instructions_dir "$ROOT/instructions" --model_name_or_path "$MODEL"
         --context_prompt full_context --no_context_prompt no_context --max_new_tokens "$MAX_TOKENS" ${CD_EXTRA_ARGS:-})
$PY "$REPO/scripts/run_contrastive_decoding.py" "${CD_ARGS[@]}" "${EVAL_FLAG[@]}" --non_context_weight 0 --save_output "$CD_DIR/c1_nc0"
$PY "$REPO/scripts/run_contrastive_decoding.py" "${CD_ARGS[@]}" "${EVAL_FLAG[@]}" --context_weight 0 --save_output "$CD_DIR/c0_nc1"
$PY "$REPO/scripts/run_contrastive_decoding.py" "${CD_ARGS[@]}" "${EVAL_FLAG[@]}" --save_output "$CD_DIR/c1_nc1"
$PY "$REPO/scripts/run_contrastive_decoding.py" "${CD_ARGS[@]}" "${EVAL_FLAG[@]}" --context_weight 5 --save_output "$CD_DIR/c5_nc1"
$PY "$REPO/scripts/run_contrastive_decoding.py" "${CD_ARGS[@]}" --num_return_sequences 4 --sample --temperature 0.7 --min_p 0.02 \
    --save_output "$CD_DIR/c1_nc1_t0.7_minp0.02_n4"

# 8. P-CXMI ------------------------------------------------------------------------------------
step "8. P-CXMI"
$PY "$REPO/scripts/pcxmi.py" --root_dir "$ROOT" --model "$MODEL" --model_stem "$MODEL_NAME" --datasets "$DS" --lps $LPS \
    --context_settings full_context no_context --targets ref hyp src --ref_source raw_data \
    --sep_tokens "$SEP_TOKENS" --backend "$BACKEND" --max_tokens "$MAX_TOKENS"
$PY "$REPO/scripts/pcxmi_hyps.py" --root_dir "$ROOT" --model "$MODEL" --model_stem "$MODEL_NAME" --gen_model_name "$MODEL_NAME" \
    --datasets "$DS" --lps $LPS --full_context_prompt_name full_context --no_context_prompt_name no_context \
    --sep_tokens "$SEP_TOKENS" --backend "$BACKEND" --gen_backend "$BACKEND"
$PY "$REPO/scripts/pcxmi_summary.py" --root_dir "$ROOT" --datasets "$DS" --lps $LPS --model_stem "$MODEL_NAME" \
    --out "$ROOT/pcxmi_summary.json" > /dev/null

# 9. report ------------------------------------------------------------------------------------
step "9. report"
$PY "$REPO/scripts/zh_pipeline_report.py" --root_dir "$ROOT" --data_name "$DATA_NAME" --split "$SPLIT" \
    --model_name "$MODEL_NAME" --gen_backend "$BACKEND" --out "$ROOT/pipeline_report.json" > /dev/null
echo "### done: $ROOT/pipeline_report.json"
