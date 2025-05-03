#!/bin/bash

llmrl_root_dir=$1

llm_model_name="TheBloke/deepseek-llm-7b-chat-GPTQ" # 
use_chatgpt=0
# --use_chatgpt
# --full_map_observable
n_episodes_to_collect=100
map_names=("2x2" "3x3" "4x4" "8x8")
script="${llmrl_root_dir}/slurm_job_trajs_collection_using_llm.bash"
for map_name in "${map_names[@]}"
do
    config=" ${map_name} ${llm_model_name} ${n_episodes_to_collect}"
    echo -e "Collecting the LLM trajectories under the config: ${config}"
    # basic prompt: partially observable
    sbatch ${script} ${config} 0 ${use_chatgpt}
    # basic + full map 1: fully obserable
    sbatch ${script} ${config} 1 ${use_chatgpt}
done

