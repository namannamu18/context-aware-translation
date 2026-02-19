#!/bin/bash

split=$1
GEN_DIR=contrast_decode/

mkdir -p ${GEN_DIR}/wmt24_chat_${split}.${lp}

export CUDA_VISIBLE_DEVICES=7

for lp in en-de en-fr en-ko en-nl en-pt; do
    python scripts/run_contrastive_decoding.py --lang_pair ${lp} --split $split --num_return_sequences 100 --sample  \
        --save_output ${GEN_DIR}/wmt24_chat_${split}.${lp}/c1_nc1_t0.7_minp0.02_n100 --temperature 0.7 --min_p 0.02 

    python scripts/run_contrastive_decoding.py --lang_pair ${lp} --split $split --non_context_weight 0 \
    --save_output ${GEN_DIR}/wmt24_chat_${split}.${lp}/c1_nc0 

    python scripts/run_contrastive_decoding.py --lang_pair ${lp} --split $split  --context_weight 0 \
    --save_output ${GEN_DIR}/wmt24_chat_${split}.${lp}/c0_nc1
        
    python scripts/run_contrastive_decoding.py --lang_pair ${lp} --split $split \
    --save_output ${GEN_DIR}/wmt24_chat_${split}.${lp}/c1_nc1

    python scripts/run_contrastive_decoding.py --lang_pair ${lp} --split $split --context_weight 5 \
    --save_output ${GEN_DIR}/wmt24_chat_${split}.${lp}/c5_nc1

    python scripts/run_contrastive_decoding.py --lang_pair ${lp} --split $split --num_beams 5 \
    --save_output ${GEN_DIR}/wmt24_chat_${split}.${lp}/c1_nc1_b5
done;