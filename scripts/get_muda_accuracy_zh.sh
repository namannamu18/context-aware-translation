#!/bin/bash
# MuDA (context-dependent phenomena) accuracy for en->zh outputs.
# MuDA (https://github.com/CoderPat/MuDA) has a Chinese tagger (spacy-stanza "zh"); it needs the stanza zh
# models and bert-base-multilingual-cased (awesome-align) from the HuggingFace Hub.
# Input: any csv with source_language/source/reference/doc_id + one column per system,
# e.g. the MBR outputs (column "output-select") or paper_results_zh/<data>/<split>.en-zh.csv with added columns.
export MUDA_HOME=${MUDA_HOME:?set MUDA_HOME to a MuDA checkout}
input_csv=${1:?input csv}
model=${2:-output-select}
out_prefix=${3:-muda_zh}
python scripts/get_muda_accuracy.py \
    --input_csv "$input_csv" \
    --model "$model" \
    --tgt-lang zh \
    --dump_hyps_tags_file ${out_prefix}_${model}_hyps_tags.json \
    --dump_refs_tags_file ${out_prefix}_${model}_refs_tags.json \
    --dump_stats_file ${out_prefix}_${model}_stats.jsonl
