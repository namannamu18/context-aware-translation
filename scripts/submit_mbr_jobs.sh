#!/bin/bash

#SBATCH --job-name=mbr-chat   # Job name
#SBATCH --time=1-00:00:00         # Run time (hh:mm:ss) 
#SBATCH --gres=gpu:1           # Number of GPUs to be used
#SBATCH --qos=gpu-medium         # QOS to be used
#SBATCH --partition=a6000         # QOS to be used
#SBATCH --output=./logs/job.%A-%a.out

module purge
module load cuda openjdk python/3.10.14

split=$1
GEN_DIR=mbr_outputs/mbr_outputs_${split}/
source /mnt/home/sweta/envs/chat-qe-env/bin/activate

data=wmt24_chat
context=full_context_empty_sys
model=TowerInstruct-7B-w-chat-empty-sys
OUT_DIR=${GEN_DIR}/${context}/${model}

context_sizes=( 0 2 6 10 15)
batch_sizes=( 512 256 64 32 16)

# for lp in  en-de en-ko en-nl en-pt en-fr ; do

#     mkdir -p $OUT_DIR/${data}_${split}.${lp}
        
#     # python scripts/run_context_comet_mbr.py --lang_pair ${lp} --use_candidates --split $split --inf_context ${context} \
#     #     --save_output $OUT_DIR/${data}_${split}.${lp}/comet_eps_w${context_size} --model_name ${model} --data_name $data --context_size ${context_size}

    
#     for i in "${!context_sizes[@]}"; do
            # context_size=${context_sizes[$i]}
            # batch_size=${batch_sizes[$i]}

#         python scripts/run_context_comet_mbr.py --lang_pair ${lp} --use_candidates --split $split \
#         --context_source "source" --context_mt "source" \
#         --inf_context ${context} --data_name $data --context_size ${context_size} \
#         --save_output $OUT_DIR/${data}_${split}.${lp}/comet_eps_context_source_w${context_size} \
#          --use_context --model_name ${model} --batch_size 16 

#         python scripts/run_context_comet_mbr.py --lang_pair ${lp} --use_candidates \
#         --context_source "source" --context_mt "comet-best" --split $split \
#             --inf_context ${context} --data_name $data  --context_size ${context_size} \
#             --save_output ${GEN_DIR}/${context}/${model}/${data}_${split}.${lp}/comet_eps_context_comet-best_w${context_size} \
#             --use_context --model_name ${model} --batch_size 16 
#     done;
# done;

context=no_context
model=TowerInstruct-7B-v0.2
OUT_DIR=${GEN_DIR}/${context}/${model}
context_size=0
batch_size=512

for lp in en-pt en-fr ; do

    mkdir -p $OUT_DIR/${data}_${split}.${lp}
        
    # python scripts/run_context_comet_mbr.py --lang_pair ${lp} --use_candidates --split $split --inf_context ${context} \
    #     --save_output $OUT_DIR/${data}_${split}.${lp}/comet_eps_w${context_size} --model_name ${model} --data_name $data --context_size ${context_size}
    
    # for i in "${!context_sizes[@]}"; do
    
        # context_size=${context_sizes[$i]}
        # batch_size=${batch_sizes[$i]}

        python scripts/run_context_comet_mbr.py --lang_pair ${lp} --use_candidates --split $split \
        --context_source "source" --context_mt "source" \
        --inf_context ${context} --data_name $data --context_size ${context_size} \
        --save_output $OUT_DIR/${data}_${split}.${lp}/comet_eps_context_source_w${context_size} \
         --use_context --model_name ${model} --batch_size ${batch_size} 

        python scripts/run_context_comet_mbr.py --lang_pair ${lp} --use_candidates \
        --context_source "source" --context_mt "comet-best" --split $split \
            --inf_context ${context} --data_name $data  --context_size ${context_size} \
            --save_output ${GEN_DIR}/${context}/${model}/${data}_${split}.${lp}/comet_eps_context_comet-best_w${context_size} \
            --use_context --model_name ${model} --batch_size ${batch_size} 
    # done;
done;