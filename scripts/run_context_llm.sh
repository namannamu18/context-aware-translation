datasets=(wmt24_chat bcontrast)

#models=("greedy-7b-ft-w-context" "greedy-7b-ft-wo-context" "greedy-7b-w-context" "greedy-7b-wo-context")
models=("greedy-7b-w-context")

for dataset in ${datasets[@]}; do
    echo "Running $dataset"
    if [ $dataset == "wmt24_chat" ]; then
        lps=(pt fr de nl ko)
    else
        lps=(de)
    fi
    for lp in ${lps[@]}; do
        for model in ${models[@]}; do
            echo "Running $model"
            python run_context_llm.py \
                --lp $lp \
                --dataset $dataset \
                --tgt_col $model \
                #--fixed_ende_examples
        done
    done
done