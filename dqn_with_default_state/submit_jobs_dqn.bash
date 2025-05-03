#!/bin/bash

map_names=(2x2 3x3 4x4 8x8)
#llm_model_name="deepseek-ai/deepseek-llm-7b-chat"
for map_name in "${map_names[@]}"
do
    echo "===> evaluate the DQN agent for the map ${map_name}"
    sbatch ./slurm_job_dqn.bash ${map_name}

done


