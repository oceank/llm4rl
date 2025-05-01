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


# run experiments
map_name="3x3"
max_steps=100000 #20000
eval_interval=1000
#seed=42
buffer_paths={1: "Map3x3_seed1_deepseek-llm-7b-chat-GPTQ_HWS5_PO_NotChatGPT_NoFewshot_NoThought_100Episodes_520Trans_0.17AveR.pkl"}

seeds=[1] #, 2, 3, 4, 5]
all_eval_returns = {}
for seed in seeds:
    if buffer_paths:
        buffer_path = buffer_paths[seed]
    else:
        buffer_path = ""
    eval_returns = train_dqn(
        env_name="FrozenLake-v1", map_name=map_name,
        max_steps=max_steps, eval_interval=eval_interval, seed=seed,
        buffer_path=buffer_path)
    all_eval_returns[seed] = eval_returns

'''
label = f"Map{map_name}"
if buffer_paths:
    label += "_withLLMTrajs"
color="blue"
save_path=f"{label}_MaxSteps{max_steps}_EvalInterval{eval_interval}.png"
plot_mean_sem(all_eval_returns, label=label, color=color, save_path=save_path)

# Save to file
with open(save_path[:-4]+".pkl", "wb") as f:
    pickle.dump(all_eval_returns, f)
'''
