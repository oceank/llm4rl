import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from dqn import train_dqn
from utility import plot_mean_sem
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--map_size", type=str, required=True, help="e.g., 2x2, 3x3, 4x4")
parser.add_argument("--observability", type=str, required=True, help="full, partial")
parser.add_argument("--llm_model_name", type=str, default="deepseek-ai/deepseek-llm-7b-chat-GPTQ")
parser.add_argument("--use_prior_trajs", action='store_true')
parser.add_argument("--buffer_dir", type=str, default="/work/suj/llm4rl/llm_agent_trajs")
args = parser.parse_args()


def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def fetch_prior_buffer_paths(map_size, observability, seeds, buffer_dir):
    buffer_filepaths = {seed:"" for seed in seeds}
    if observability == "full":
        obs_tag = "FO1"
    elif observability == "partial":
        obs_tag = "PO"
    else:
        raise f"unsupported {observability}"

    all_buffer_filenames = list_files(buffer_dir)
    for seed in seeds:
        for fn in all_buffer_filenames:
            if (f"seed{seed}" in fn) and (obs_tag in fn) and (f"Map{map_size}" in fn):
                buffer_filepaths[seed] = os.path.join(buffer_dir, fn)
    return buffer_filepaths

# run experiments
map_name=args.map_size #"3x3"
observability=args.observability
max_steps=100000 #20000
eval_interval=1000
seeds=[1, 2, 3, 4, 5]
buffer_paths=None
if args.use_prior_trajs:
    buffer_paths = fetch_prior_buffer_paths(args.map_size, args.observability, seeds, args.buffer_dir)


all_eval_returns = {}
for seed in seeds:
    if buffer_paths:
        buffer_path = buffer_paths[seed]
    else:
        buffer_path = ""
    eval_returns = train_dqn(
        env_name="FrozenLake-v1", map_name=map_name,
        max_steps=max_steps, eval_interval=eval_interval, seed=seed,
        buffer_path=buffer_path,
        observability=observability)
    all_eval_returns[seed] = eval_returns


label = f"Map{map_name}"
if buffer_paths:
    label += "_withLLMTrajs"
color="blue"
save_path=f"{label}_MaxSteps{max_steps}_EvalInterval{eval_interval}_{observability}Obs"
if args.use_prior_trajs:
    save_path += f"_priorTrajs"
save_path += ".png"

plot_mean_sem(all_eval_returns, label=label, color=color, save_path=save_path)

# Save to file
with open(save_path[:-4]+".pkl", "wb") as f:
    pickle.dump(all_eval_returns, f)


