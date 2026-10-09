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
#   PROMPT_SUFFIX=               "" (base model: no_context / full_context prompts) or "_empty_sys" (the paper's fine-tuned chat
#                                model: no_context_empty_sys / full_context_empty_sys prompts)
#   STAGES=all                   or a subset of: data instructions greedy eval candidates mbr cd pcxmi report
#   CD_VARIANTS="c1_nc0 c0_nc1 c1_nc1 c5_nc1 c1_nc1_t0.7_minp0.02_n4"   contrastive decoding settings (one model load for all)
#   MBR_STYLES="source comet-best"   context-aware COMET MBR variants per context size (the paper's primary system is "source", w=2)
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
PROMPT_SUFFIX=${PROMPT_SUFFIX:-}
STAGES=${STAGES:-all}
CD_VARIANTS=${CD_VARIANTS:-"c1_nc0 c0_nc1 c1_nc1 c5_nc1 c1_nc1_t0.7_minp0.02_n4"}
MBR_STYLES=${MBR_STYLES:-"source comet-best"}
NC=no_context${PROMPT_SUFFIX}
FC=full_context${PROMPT_SUFFIX}
DS=${DATA_NAME}_${SPLIT}
LPS="en-zh zh-en"

want() { [ "$STAGES" = "all" ] || [[ " $STAGES " == *" $1 "* ]]; }

mkdir -p "$ROOT"
ROOT=$(cd "$ROOT" && pwd)
echo "### run_zh_pipeline: ROOT=$ROOT DS=$DS MODEL=$MODEL BACKEND=$BACKEND N=$N_CANDIDATES UTILITY=$UTILITY PROMPTS=$FC/$NC STAGES=$STAGES"

step() { echo; echo "=================== $* ==================="; }

# 1. data -----------------------------------------------------------------------------------
step "1. data"
if ! want data; then echo "(skipped)"
elif [ -n "$MAX_DOCS" ]; then
    $PY "$REPO/scripts/subset_zh_dataset.py" --src_root "$REPO" --dst_root "$ROOT" --data_name "$DATA_NAME" \
        --split "$SPLIT" --max_docs "$MAX_DOCS" --skip_docs "$SKIP_DOCS"
elif [ "$ROOT" != "$REPO" ]; then
    $PY "$REPO/scripts/subset_zh_dataset.py" --src_root "$REPO" --dst_root "$ROOT" --data_name "$DATA_NAME" --split "$SPLIT"
fi

# 2. instructions ---------------------------------------------------------------------------
step "2. instructions"
if want instructions; then
$PY "$REPO/scripts/make_instructions.py" --root_dir "$ROOT" --datasets "$DS" --pairs en-zh \
    --conditions no_context full_context no_context_empty_sys full_context_empty_sys full_context_6_turns
fi

# 3. greedy decoding (tower-eval gen equivalent) + 4. evaluation ------------------------------
step "3. greedy decoding ($NC / $FC)"
if want greedy; then
$PY "$REPO/scripts/generate_translations.py" --root_dir "$ROOT" --model "$MODEL" --model_name "$MODEL_NAME" \
    --conditions $NC $FC --datasets "$DS" --lps $LPS --backend "$BACKEND" --max_tokens "$MAX_TOKENS" \
    --repetition_penalty "$REPETITION_PENALTY" --loop_retry_penalty "$LOOP_RETRY_PENALTY"
fi
step "4. evaluation"
if want eval; then
$PY "$REPO/scripts/evaluate_translations.py" --root_dir "$ROOT" --model_name "$MODEL_NAME" --backend "$BACKEND" \
    --conditions $NC $FC --datasets "$DS" --lps $LPS --comet_model "$COMET_MODEL"
fi

# 5. candidates (epsilon sampling: temperature 0.7, min_p 0.02) ------------------------------
step "5. epsilon-sampling candidates"
if want candidates; then
$PY "$REPO/scripts/generate_candidates.py" --root_dir "$ROOT" --model "$MODEL" --model_stem "$MODEL_NAME" \
    --datasets "$DS" --lps $LPS --prompts $NC $FC --n_candidates "$N_CANDIDATES" \
    --backend "$BACKEND" --max_tokens "$MAX_TOKENS"
fi

# 6. MBR / QAD --------------------------------------------------------------------------------
step "6. MBR decoding"
if want mbr; then
cd "$ROOT"  # run_context_comet_mbr.py reads candidates/ and generations/ relative to the cwd
for context in $NC $FC; do
    OUT_DIR=mbr_outputs/mbr_outputs_${SPLIT}/${context}/${MODEL_NAME}/${DS}.en-zh
    mkdir -p "$OUT_DIR"
    MBR_ARGS=(--lang_pair en-zh --split "$SPLIT" --data_name "$DATA_NAME" --data_dir "$ROOT/paper_results_zh"
              --inf_context "$context" --model_name "$MODEL_NAME" --num_candidates "$N_CANDIDATES"
              --gen_backend "$BACKEND" --comet_model "$COMET_MODEL" --utility "$UTILITY" --batch_size 16)
    # (a) standard (source-only) COMET MBR
    $PY "$REPO/scripts/run_context_comet_mbr.py" "${MBR_ARGS[@]}" --save_output "$OUT_DIR/comet_eps"
    for w in $CONTEXT_SIZES; do
        for style in $MBR_STYLES; do
            # (b) context-aware COMET MBR, context = previous source utterances (style "source", the paper's primary system)
            # (c) context-aware COMET MBR, context = COMET-MBR-selected translations of previous utterances (style "comet-best")
            $PY "$REPO/scripts/run_context_comet_mbr.py" "${MBR_ARGS[@]}" --use_context --context_size "$w" \
                --context_source source --context_mt "$style" --save_output "$OUT_DIR/comet_eps_context_${style}_w${w}"
        done
    done
done
cd - > /dev/null
fi

# 7. contrastive decoding (context vs. no-context prompts) ------------------------------------
step "7. contrastive decoding"
if want cd; then
CD_DIR=$ROOT/contrast_decode/${DS}.en-zh
mkdir -p "$CD_DIR"
EVAL_FLAG=()
[ "$UTILITY" = "comet" ] && EVAL_FLAG=(--eval --comet_model "$COMET_MODEL")
CD_ARGS=(--lang_pair en-zh --split "$SPLIT" --data_name "$DATA_NAME" --data_dir "$ROOT/paper_results_zh/$DATA_NAME"
         --instructions_dir "$ROOT/instructions" --model_name_or_path "$MODEL"
         --context_prompt "$FC" --no_context_prompt "$NC" --max_new_tokens "$MAX_TOKENS" ${CD_EXTRA_ARGS:-})
# all variants in one process: the model (and COMET) is loaded once
$PY "$REPO/scripts/run_contrastive_decoding.py" "${CD_ARGS[@]}" "${EVAL_FLAG[@]}" --variants $CD_VARIANTS --save_dir "$CD_DIR"
fi

# 8. P-CXMI ------------------------------------------------------------------------------------
step "8. P-CXMI"
if want pcxmi; then
$PY "$REPO/scripts/pcxmi.py" --root_dir "$ROOT" --model "$MODEL" --model_stem "$MODEL_NAME" --datasets "$DS" --lps $LPS \
    --context_settings $FC $NC --targets ref hyp src --ref_source raw_data \
    --sep_tokens "$SEP_TOKENS" --backend "$BACKEND" --max_tokens "$MAX_TOKENS"
$PY "$REPO/scripts/pcxmi_hyps.py" --root_dir "$ROOT" --model "$MODEL" --model_stem "$MODEL_NAME" --gen_model_name "$MODEL_NAME" \
    --datasets "$DS" --lps $LPS --full_context_prompt_name "$FC" --no_context_prompt_name "$NC" \
    --sep_tokens "$SEP_TOKENS" --backend "$BACKEND" --gen_backend "$BACKEND"
$PY "$REPO/scripts/pcxmi_summary.py" --root_dir "$ROOT" --datasets "$DS" --lps $LPS --model_stem "$MODEL_NAME" \
    --out "$ROOT/pcxmi_summary.json" > /dev/null
fi

# 9. report ------------------------------------------------------------------------------------------------------
step "9. report"
if want report; then
$PY "$REPO/scripts/zh_pipeline_report.py" --root_dir "$ROOT" --data_name "$DATA_NAME" --split "$SPLIT" \
    --model_name "$MODEL_NAME" --gen_backend "$BACKEND" --out "$ROOT/pipeline_report.json" > /dev/null
echo "### done: $ROOT/pipeline_report.json"
fi
