#!/bin/bash

#SBATCH --job-name=eval-muda   # Job name
#SBATCH --time=2-00:00:00         # Run time (hh:mm:ss) 
#SBATCH --gres=gpu:1            # Number of GPUs to be used
#SBATCH --qos=gpu-long         # QOS to be used
#SBATCH --nodelist=artemis        # Node to use


# source ~/.bashrc
export CUDA_VISIBLE_DEVICES=7
export MUDA_HOME=/mnt/data-poseidon/sweta/context-aware-mt/chat-translation-generation/wmt24-chat-translation/metrics/MuDA
# source /mnt/data-poseidon/sweta/chat-translation-generation/wmt24-chat-translation/muda-env/bin/activate

tgt_langs=(de)
# models=(greedy-7b-w-context greedy-7b-wo-context greedy-7b-ft-w-context greedy-7b-ft-wo-context mbr mbr-source mbr-instruct mbr-source-instruct)
models=(towerchat-ft-7b-w-context towerchat-ft-7b-wo-context)
split=test
data=bcontrast

for tgt_lang in ${tgt_langs[@]}; do
    for model in ${models[@]}; do
        echo "tgt_lang: $tgt_lang, Model: $model"
        python scripts/get_muda_accuracy.py \
            --input_csv tacl_review/${data}/${split}.en-${tgt_lang}.csv \
            --model $model \
            --tgt-lang $tgt_lang \
            --dump_hyps_tags_file tacl_review/${data}/${split}_${tgt_lang}_${model}_hyps_tags.json \
            --dump_refs_tags_file tacl_review/${data}/${split}_${tgt_lang}_${model}_refs_tags.json \
            --dump_stats_file tacl_review/${data}/${split}_${tgt_lang}_${model}_stats.jsonl \
            --awesome-align-cachedir /mnt/scratch-artemis/sweta/cache
    done
done
