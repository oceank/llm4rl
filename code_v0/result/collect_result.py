#!/bin/bash

import os
import json
import numpy as np
import scipy
import scipy.stats as stats

result_dir_root="./" # this script should be placed at the root of the result folder

env_name="FrozenLake-v1"
map_name="map3x3_NotSlippery" #"map2x2OneStep_NotSlippery"
env_id=f"{env_name}_{map_name}"


llm_model_name="TheBloke/deepseek-llm-7b-chat-GPTQ"
agent_name="LLM"
h_tag="H0" #0, 5
o_tag="FM2" # FM0, FM1, FM2, PM
prompt_tag=f"basic_{h_tag}_{o_tag}"
use_milestones=False
if use_milestones:
    prompt_tag += "_UseMilestones"

train_eval_tag="EvalEp25EvalIntv25TrainEp15000"

experiment_dir=os.path.join(result_dir_root, env_id, f"{llm_model_name}_{agent_name}", prompt_tag, train_eval_tag)

seeds=[1,2,3,4,5]
eval_rts=[]
for seed in seeds:
    fp = os.path.join(experiment_dir, f"seed_{seed}/results.json")
    with open(fp) as file:
        data = json.load(file)
        rt = data.get("eval_ave_return")
        eval_rts.append(rt)
eval_rts = np.array(eval_rts)
eval_rt_mean = np.mean(eval_rts)
eval_rt_sem = stats.sem(eval_rts)
print(f"[{env_id}, {llm_model_name}, {prompt_tag}, {train_eval_tag}]:\n\t{eval_rt_mean}, {eval_rt_sem}")
