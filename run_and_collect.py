import os
import subprocess
import json
import numpy as np
from scipy import stats

# ========== CONFIGURATION ==========
map_names = ["2x2"]
seeds = [1, 2, 3, 4, 5]

# Training & Evaluation Settings
n_episodes = 10000
eval_interval = 25
n_episodes_eval = 25
max_new_tokens = 32 #128
history_window_size = 0 # 0, 5
full_map_observable = False
full_map_desc_type = 0 # 0, 1, 2

# LLM Info
llm_model_name = "TheBloke/deepseek-llm-7b-chat-GPTQ"
agent_name = "LLM" # RND, RL
use_chatgpt = False
use_fewshot = False

# File paths
toy_agent_script = "./toy_agent.py"  # Update if needed
result_root = "./result"

# ========== MAIN LOOP ==========
for map_name in map_names:
    env_name = "FrozenLake-v1"
    env_id = f"{env_name}_map{map_name}_NotSlippery"

    # Create prompt and training tags
    h_tag = f"H{history_window_size}"
    o_tag = f"FM{full_map_desc_type}" if full_map_observable else "PM"
    prompt_tag = f"basic_{h_tag}_{o_tag}"
    if use_fewshot:
        prompt_tag = f"Fewshot_{prompt_tag}"
    else:
        prompt_tag = f"Zeroshot_{prompt_tag}"
    train_eval_tag = f"EvalEp{n_episodes_eval}EvalIntv{eval_interval}TrainEp{n_episodes}"

    experiment_dir = os.path.join(
        result_root, env_id, f"{llm_model_name}_{agent_name}", prompt_tag, train_eval_tag
    )

    # === Run experiments ===
    for seed in seeds:
        print(f"=== Running experiment: map={map_name}, seed={seed} ===")
        cmd = [
            "python", toy_agent_script,
            "--update_type", "llm",
            "--seed", str(seed),
            "--map_name", map_name,
            "--max_new_tokens", str(max_new_tokens),
            "--history_window_size", str(history_window_size),
            "--n_episodes_eval", str(n_episodes_eval),
            "--n_episodes", str(n_episodes),
            "--eval_interval", str(eval_interval),
            "--llm_model_name", llm_model_name,
        ]
        if full_map_observable:
            cmd += ["--full_map_observable", "--full_map_desc_type", str(full_map_desc_type)]
        if use_chatgpt:
            cmd += ["--use_chatgpt"]
        if use_fewshot:
            cmd += ["--use_fewshot"]

        subprocess.run(cmd, check=True)

    # === Collect results ===
    eval_returns = []
    for seed in seeds:
        result_file = os.path.join(experiment_dir, f"seed_{seed}", "results.json")
        if not os.path.isfile(result_file):
            print(f"[Warning] Missing result for seed {seed}")
            continue

        with open(result_file, "r") as f:
            result = json.load(f)
            rt = result.get("eval_ave_return")
            eval_returns.append(rt)

    # === Aggregate stats ===
    if eval_returns:
        eval_returns = np.array(eval_returns)
        mean_return = np.mean(eval_returns)
        sem_return = stats.sem(eval_returns)
        print(f"\n[{env_id}, {llm_model_name}, {prompt_tag}, {train_eval_tag}]")
        print(f"Mean eval return: {mean_return:.4f}, SEM: {sem_return:.4f}\n")
    else:
        print(f"\n[Error] No valid results found for map={map_name}\n")

