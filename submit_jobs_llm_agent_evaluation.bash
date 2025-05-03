#!/bin/bash

llmrl_root_dir=$1

map_names=(2x2 3x3 4x4 8x8)
llm_model_name="deepseek-ai/deepseek-llm-7b-chat"
for map_name in "${map_names[@]}"
do
    echo "===> evaluate the LLM agent (${llm_model_name}) for the map ${map_name}"
    sbatch ./slurm_job_llm_agent_evaluation.bash ${llmrl_root_dir} ${map_name} ${llm_model_name}

done


