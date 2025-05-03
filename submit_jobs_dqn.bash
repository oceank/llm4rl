#!/bin/bash


buffer_dir="/work/suj/llm4rl/llm_agent_trajs"
map_names=(2x2 3x3 4x4 8x8)
observabilities=("partial" "full")
#llm_model_name="deepseek-ai/deepseek-llm-7b-chat"
buffer_dirs=("" "/work/suj/llm4rl/llm_agent_trajs")
for map_name in "${map_names[@]}"
do
    for observability in "${observabilities[@]}"
    do
        for buffer_dir in "${buffer_dirs[@]}"
        do
            echo "===> evaluate the DQN agent for the map ${map_name} with ${observability} observable 3-D observation and possible prior buffer in ('${buffer_dir}')"
            sbatch ./slurm_job_dqn.bash "${map_name}" "${observability}" "${buffer_dir}"
        done
    done

done


