#!/bin/bash

#SBATCH --job-name=mbr-chat   # Job name
#SBATCH --time=24:00:00         # Run time (hh:mm:ss) 
#SBATCH --gres=gpu:1           # Number of GPUs to be used
#SBATCH --qos=gpu-long         # QOS to be used
#SBATCH --partition=a6000         # QOS to be used
#SBATCH --output=./logs/job.%A-%a.out

# source ~/.bashrc
export VLLM_LOGGING_LEVEL=DEBUG

module purge
module load cuda openjdk python/3.10.14

source /mnt/home/sweta/envs/chat-qe-env/bin/activate


split=$1
GEN_DIR=mbr_outputs/mbr_outputs_${split}/

lp="en-de"
data=bcontrast

context=no_context
model=TowerInstruct-7B-v0.2

mkdir -p ${GEN_DIR}/${context}/${model}/${data}_${split}.${lp}

# python scripts/run_context_comet_mbr.py --lang_pair ${lp} --use_candidates \
#     --inf_context ${context} --batch_size 16  --split $split  \
#     --save_output ${GEN_DIR}/${context}/${model}/${data}_${split}.${lp}/comet_eps \
#     --model_name ${model} --data_name $data

for context_size in 10 15; do
    python scripts/run_context_comet_mbr.py --lang_pair ${lp} --use_candidates --split $split \
        --context_source "source" --context_mt "source" \
        --inf_context ${context} --data_name $data --context_size ${context_size} \
        --save_output ${GEN_DIR}/${context}/${model}/${data}_${split}.${lp}/comet_eps_context_source_w${context_size} \
        --use_context --model_name ${model} --batch_size 16 

    python scripts/run_context_comet_mbr.py --lang_pair ${lp} --use_candidates \
    --context_source "source" --context_mt "comet-best" --split $split \
        --inf_context ${context} --data_name $data --context_size ${context_size} \
        --save_output ${GEN_DIR}/${context}/${model}/${data}_${split}.${lp}/comet_eps_context_comet-best_w${context_size} \
        --use_context --model_name ${model} --batch_size 16 
done;

context=full_context_empty_sys
model=TowerInstruct-7B-w-chat-empty-sys

mkdir -p ${GEN_DIR}/${context}/${model}/${data}_${split}.${lp}

# python scripts/run_context_comet_mbr.py --lang_pair ${lp} --use_candidates \
#     --inf_context ${context} --batch_size 16  --split $split  \
#     --save_output ${GEN_DIR}/${context}/${model}/${data}_${split}.${lp}/comet_eps \
#     --model_name ${model} --data_name $data

for context_size in 0 2 6 10 15; do
    python scripts/run_context_comet_mbr.py --lang_pair ${lp} --use_candidates --split $split \
        --context_source "source" --context_mt "source" \
        --inf_context ${context} --data_name $data --context_size ${context_size} \
        --save_output ${GEN_DIR}/${context}/${model}/${data}_${split}.${lp}/comet_eps_context_source_w${context_size} \
        --use_context --model_name ${model} --batch_size 16 

    python scripts/run_context_comet_mbr.py --lang_pair ${lp} --use_candidates \
    --context_source "source" --context_mt "comet-best" --split $split \
        --inf_context ${context} --data_name $data --context_size ${context_size}  \
        --save_output ${GEN_DIR}/${context}/${model}/${data}_${split}.${lp}/comet_eps_context_comet-best_w${context_size} \
        --use_context --model_name ${model} --batch_size 16 
done;

# export CUDA_VISIBLE_DEVICES=7
# export MUDA_HOME=/mnt/data/sagrawal/wmt24-chat-translation/MuDA

# tgt_lang=de
# models=(greedy-7b-w-context greedy-7b-wo-context greedy-7b-ft-w-context greedy-7b-ft-wo-context mbr-instruct mbr-source-instruct mbr-chat mbr-source-chat)
# split=test
# data=bcontrast

# for model in ${models[@]}; do
#     echo "tgt_lang: $tgt_lang, Model: $model"
#     python scripts/get_muda_accuracy.py \
#         --input_csv tacl_results/${data}/${split}.en-${tgt_lang}.csv \
#         --model $model \
#         --tgt-lang $tgt_lang \
#         --dump_hyps_tags_file tacl_results/${data}/${split}_${tgt_lang}_${model}_hyps_tags.json \
#         --dump_refs_tags_file tacl_results/${data}/${split}_${tgt_lang}_${model}_refs_tags.json \
#         --dump_stats_file tacl_results/${data}/${split}_${tgt_lang}_${model}_stats.jsonl \
#         --awesome-align-cachedir /mnt/data/sagrawal/cache
# done