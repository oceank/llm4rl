#!/bin/bash


root_dir=$1
llm_model_name="TheBloke/deepseek-llm-7b-chat-GPTQ" # 
# --use_chatgpt
# --full_map_observable
n_episodes_to_collect=100
map_names=("2x2" "3x3" "4x4" "8x8")
script="${root_dir}/run_collect_llm_experiences_v2.py"
for map_name in "${map_names[@]}"
do
    config="--llm_model_name ${llm_model_name} --map_name ${map_name} --n_episodes_to_collect ${n_episodes_to_collect}"
    # basic prompt: partially observable
    python ${script} ${config}
    # basic + full map 1: fully obserable
    python ${script} ${config} --full_map_observable
done

