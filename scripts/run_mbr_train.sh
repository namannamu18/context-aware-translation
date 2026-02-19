#!/bin/bash

#SBATCH --job-name=mbr-chat   # Job name
#SBATCH --time=2-00:00:00         # Run time (hh:mm:ss) 
#SBATCH --gres=gpu:1           # Number of GPUs to be used
#SBATCH --qos=gpu-h100         # QOS to be used
#SBATCH --partition=h100         # QOS to be used
#SBATCH --output=./logs/job.%A-%a.out

# module purge
# source /mnt/home/sweta/envs/chat-qe-env/bin/activate

# source /etc/profile.d/02-lmod.sh
# source ~/.bashrc
# module load cuda openjdk python/3.10.14
# source ~/envs/chat-qe-env/bin/activate

split=train
GEN_DIR=mbr_outputs/mbr_outputs_${split}/


data=wmt24_chat
context=full_context_empty_sys
model=TowerInstruct-7B-w-chat-empty-sys
OUT_DIR=${GEN_DIR}/${context}/${model}

context_sizes=( 0 2)

for lp in en-fr ; do

    mkdir -p $OUT_DIR/${data}_${split}.${lp}
    
    for i in "${!context_sizes[@]}"; do
        echo $context_size
        context_size=${context_sizes[$i]}
        python scripts/run_context_comet_mbr.py --lang_pair ${lp} --use_candidates --split $split \
        --context_source "source" --context_mt "source" \
        --inf_context ${context} --data_name $data --context_size ${context_size} \
        --save_output $OUT_DIR/${data}_${split}.${lp}/comet_eps_context_source_w${context_size} \
         --use_context --model_name ${model} --batch_size 16 
    done;
done;

