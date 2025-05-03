#!/bin/sh
#SBATCH --job-name=llmrl
#SBATCH -N 1
#SBATCH --ntasks 1 --cpus-per-task 12 #-n 24, 16, 12   ## how many cpu cores to request
# -B 1:8:2
# --exclusive
#SBATCH --gres=gpu:1   ## Run on 1 GPU
#SBATCH --output /work/suj/llm4rl/slurm/llmrl_%j.out
#SBATCH --error /work/suj/llm4rl/slurm/llmrl_%j.err
#SBATCH -p dgx_aic,AI_Center,AI_Center_L40S,v100-32gb-hiprio,gpu-v100-32gb
#AI_Center,v100-32gb-hiprio,v100-16gb-hiprio,gpu-v100-32gb,gpu-v100-16gb
#dgx_aic,AI_Center,gpu-v100-16gb,v100-16gb-hiprio,gpu-v100-32gb,v100-32gb-hiprio

#SBATCH --exclude=node[363,372,375,383,387,397]


# Accessibale GPU queues
#   AI_Center_L40S,AI_Center,dgx_aic,gpu-v100-16gb,gpu-v100-32gb,v100-16gb-hiprio,v100-32gb-hiprio
#   AI_Center,v100-32gb-hiprio,gpu-v100-32gb
#   gpu-v100-16gb,v100-16gb-hiprio


echo "\n======= Setup the environment =======\n"
module load python3/anaconda/2021.11
module load cuda/12.1 # 12.3
source activate llmrl # llm4rl

echo "\n======= Check the environment =======\n"
echo $CONDA_PREFIX
echo $(uname -a)
nvcc --version
nvidia-smi

llmrl_root_dir=$1
map_name=$2
llm_model_name=$3


script="${evaluate_LLM_agent.py}/evaluate_LLM_agent.py"

echo "===> evaluate the LLM agent (${llm_model_name}) for the map ${map_name}"
python ${script} --map_size ${map_name} --llm_model_name ${llm_model_name}

