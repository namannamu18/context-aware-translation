#!/bin/bash
source ~/venvs/venv_wmt24_chat_translation/bin/activate
cd ~/wmt24_chat_MT/wmt24-chat-translation
export PYTHONPATH=./

##en-de
#python alti_analysis/aggregate_contributions.py --model_name tower-chat --model_prec fp16 --contributions_out_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/en-de/greedy-7b-ft-w-context_fp16_tok_370 --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/aggregated/en-de/greedy-7b-ft-w-context_fp16_tok_370/test --data_path alti_prep_data/test_en-de_datadf.json --model_version greedy-7b-ft-w-context --language en-de
#python alti_analysis/aggregate_contributions.py --model_name tower-chat --model_prec fp16 --contributions_out_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/en-de/greedy-7b-ft-wo-context_fp16_tok_370 --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/aggregated/en-de/greedy-7b-ft-wo-context_fp16_tok_370/test --data_path alti_prep_data/test_en-de_datadf.json --model_version greedy-7b-ft-wo-context --language en-de
#python alti_analysis/aggregate_contributions.py --model_name tower-instruct --model_prec fp16 --contributions_out_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/en-de/greedy-7b-w-context_fp16_tok_370 --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/aggregated/en-de/greedy-7b-w-context_fp16_tok_370/test --data_path alti_prep_data/test_en-de_datadf.json --model_version greedy-7b-w-context --language en-de
#python alti_analysis/aggregate_contributions.py --model_name tower-instruct --model_prec fp16 --contributions_out_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/en-de/greedy-7b-wo-context_fp16_tok_370 --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/aggregated/en-de/greedy-7b-wo-context_fp16_tok_370/test --data_path alti_prep_data/test_en-de_datadf.json --model_version greedy-7b-wo-context --language en-de
#
##de-en
#python alti_analysis/aggregate_contributions.py --model_name tower-chat --model_prec fp16 --contributions_out_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/de-en/greedy-7b-ft-w-context_fp16_tok_370 --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/aggregated/de-en/greedy-7b-ft-w-context_fp16_tok_370/test --data_path alti_prep_data/test_de-en_datadf.json --model_version greedy-7b-ft-w-context --language de-en
#python alti_analysis/aggregate_contributions.py --model_name tower-chat --model_prec fp16 --contributions_out_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/de-en/greedy-7b-ft-wo-context_fp16_tok_370 --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/aggregated/de-en/greedy-7b-ft-wo-context_fp16_tok_370/test --data_path alti_prep_data/test_de-en_datadf.json --model_version greedy-7b-ft-wo-context --language de-en
#python alti_analysis/aggregate_contributions.py --model_name tower-instruct --model_prec fp16 --contributions_out_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/de-en/greedy-7b-w-context_fp16_tok_370 --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/aggregated/de-en/greedy-7b-w-context_fp16_tok_370/test --data_path alti_prep_data/test_de-en_datadf.json --model_version greedy-7b-w-context --language de-en
#python alti_analysis/aggregate_contributions.py --model_name tower-instruct --model_prec fp16 --contributions_out_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/de-en/greedy-7b-wo-context_fp16_tok_370 --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/aggregated/de-en/greedy-7b-wo-context_fp16_tok_370/test --data_path alti_prep_data/test_de-en_datadf.json --model_version greedy-7b-wo-context --language de-en


##en-ko
#python alti_analysis/aggregate_contributions.py --model_name tower-chat --model_prec fp16 --contributions_out_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/en-ko/greedy-7b-ft-w-context_fp16_tok_370 --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/aggregated/en-ko/greedy-7b-ft-w-context_fp16_tok_370/test --data_path alti_prep_data/test_en-ko_datadf.json --model_version greedy-7b-ft-w-context --language en-ko
#python alti_analysis/aggregate_contributions.py --model_name tower-chat --model_prec fp16 --contributions_out_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/en-ko/greedy-7b-ft-wo-context_fp16_tok_370 --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/aggregated/en-ko/greedy-7b-ft-wo-context_fp16_tok_370/test --data_path alti_prep_data/test_en-ko_datadf.json --model_version greedy-7b-ft-wo-context --language en-ko
#python alti_analysis/aggregate_contributions.py --model_name tower-instruct --model_prec fp16 --contributions_out_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/en-ko/greedy-7b-w-context_fp16_tok_370 --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/aggregated/en-ko/greedy-7b-w-context_fp16_tok_370/test --data_path alti_prep_data/test_en-ko_datadf.json --model_version greedy-7b-w-context --language en-ko
#python alti_analysis/aggregate_contributions.py --model_name tower-instruct --model_prec fp16 --contributions_out_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/en-ko/greedy-7b-wo-context_fp16_tok_370 --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/aggregated/en-ko/greedy-7b-wo-context_fp16_tok_370/test --data_path alti_prep_data/test_en-ko_datadf.json --model_version greedy-7b-wo-context --language en-ko
#
##ko-en
#python alti_analysis/aggregate_contributions.py --model_name tower-chat --model_prec fp16 --contributions_out_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/ko-en/greedy-7b-ft-w-context_fp16_tok_370 --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/aggregated/ko-en/greedy-7b-ft-w-context_fp16_tok_370/test --data_path alti_prep_data/test_ko-en_datadf.json --model_version greedy-7b-ft-w-context --language ko-en
#python alti_analysis/aggregate_contributions.py --model_name tower-chat --model_prec fp16 --contributions_out_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/ko-en/greedy-7b-ft-wo-context_fp16_tok_370 --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/aggregated/ko-en/greedy-7b-ft-wo-context_fp16_tok_370/test --data_path alti_prep_data/test_ko-en_datadf.json --model_version greedy-7b-ft-wo-context --language ko-en
#python alti_analysis/aggregate_contributions.py --model_name tower-instruct --model_prec fp16 --contributions_out_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/ko-en/greedy-7b-w-context_fp16_tok_370 --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/aggregated/ko-en/greedy-7b-w-context_fp16_tok_370/test --data_path alti_prep_data/test_ko-en_datadf.json --model_version greedy-7b-w-context --language ko-en
#python alti_analysis/aggregate_contributions.py --model_name tower-instruct --model_prec fp16 --contributions_out_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/ko-en/greedy-7b-wo-context_fp16_tok_370 --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions/aggregated/ko-en/greedy-7b-wo-context_fp16_tok_370/test --data_path alti_prep_data/test_ko-en_datadf.json --model_version greedy-7b-wo-context --language ko-en



# Normalized Contributions Aggregation
#tgt_langs=( "en-de" "en-fr" "en-ko" "en-nl" "en-pt" )
#lang="en-pt"
#aggr="turn-level" #for context-level
#
##Tower Chat
#n_turns="15"
#context_type="full_context_empty_sys_10_turns"
#python alti_analysis/aggregate_contributions.py --model_name TowerInstruct-7B-w-chat-empty-sys --model_prec fp16 \
#--contributions_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions_official/$context_type/TowerInstruct-7B-w-chat-empty-sys_fp16_tok370/ \
#--save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions_official_norm_aggregated/$context_type/TowerInstruct-7B-w-chat-empty-sys_fp16_tok370/ \
#--context_aggregation_type $aggr  --num_turns $n_turns --language $lang  --normalized
#
##Tower Instruct
#n_turns="10"
#context_type="full_context_10_turns"
#python alti_analysis/aggregate_contributions.py --model_name TowerInstruct-7B-v0.2 --model_prec fp16 \
#--contributions_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions_official/$context_type/TowerInstruct-7B-v0.2_fp16_tok370/ \
#--save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions_official_norm_aggregated/$context_type/TowerInstruct-7B-v0.2_fp16_tok370/ \
#--context_aggregation_type $aggr  --num_turns $n_turns --language $lang  --normalized


# Total Contributions Aggregation
#tgt_langs=( "en-de" "en-fr" "en-ko" "en-nl" "en-pt" )

tgt_langs=( "en-de" "en-fr" "en-ko" "en-nl" "en-pt" )
for lang in "${tgt_langs[@]}"; do
    aggr="context-level" # or  context-level
    #Tower Chat
    n_turns="10"
    context_type="full_context_empty_sys_10_turns"
    python alti_analysis/aggregate_contributions.py --model_name TowerInstruct-7B-w-chat-empty-sys --model_prec fp16 \
    --contributions_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions_official/$context_type/TowerInstruct-7B-w-chat-empty-sys_fp16_tok370/ \
    --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions_official_total_aggregated/$context_type/TowerInstruct-7B-w-chat-empty-sys_fp16_tok370/ \
    --context_aggregation_type $aggr  --num_turns $n_turns --language $lang

    #Tower Instruct
    n_turns="10"
    context_type="full_context_10_turns"
    python alti_analysis/aggregate_contributions.py --model_name TowerInstruct-7B-v0.2 --model_prec fp16 \
    --contributions_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions_official/$context_type/TowerInstruct-7B-v0.2_fp16_tok370/ \
    --save_folder /mnt/scratch-artemis/manos/data/wmt24_chat_mt/alti_contributions_official_total_aggregated/$context_type/TowerInstruct-7B-v0.2_fp16_tok370/ \
    --context_aggregation_type $aggr  --num_turns $n_turns --language $lang
done