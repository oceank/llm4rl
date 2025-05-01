import os

import argparse
import subprocess
import json
import numpy as np
from scipy import stats
import pandas as pd

# ====== Parse Arguments ======
parser = argparse.ArgumentParser()
parser.add_argument("--map_size", type=str, required=True, help="e.g., 2x2, 3x3, 4x4")
parser.add_argument("--llm_model_name", type=str, required=True, help="e.g., TheBloke/deepseek-llm-7b-chat-GPTQ")
args = parser.parse_args()

map_name = args.map_size
llm_model_name = args.llm_model_name

# ====== Setup ======
#map_name = "3x3"
env_name = "FrozenLake-v1"
seeds = [1, 2, 3, 4, 5]
agent_name = "LLM"
#llm_model_name = "TheBloke/deepseek-llm-7b-chat-GPTQ"
toy_agent_script = "./toy_agent.py"  # Modify if not in current directory
result_root = "./result"
n_episodes = 10000
eval_interval = 25
n_episodes_eval = 25
max_new_tokens = 32 #128

# ====== Define Configurations ======
configurations = [
    {"desc": "basic", "history": False, "full_map": False, "full_map_type": 0, "few_shot": False},
    {"desc": "basic + history", "history": True, "full_map": False, "full_map_type": 0, "few_shot": False},
    {"desc": "basic + full map 0", "history": False, "full_map": True, "full_map_type": 0, "few_shot": False},
    {"desc": "basic + full map 1", "history": False, "full_map": True, "full_map_type": 1, "few_shot": False},
    {"desc": "basic + full map 2", "history": False, "full_map": True, "full_map_type": 2, "few_shot": False},
    {"desc": "basic + fulll map 1 + history", "history": True, "full_map": True, "full_map_type": 1, "few_shot": False},
    {"desc": "basic + few-shots", "history": False, "full_map": False, "full_map_type": 0, "few_shot": True},
    {"desc": "basic + few-shots + history", "history": True, "full_map": False, "full_map_type": 0, "few_shot": True},
    {"desc": "basic + few-shots + full map 1", "history": False, "full_map": True, "full_map_type": 1, "few_shot": True},
    {"desc": "basic + few-shots + full map 1 + history", "history": True, "full_map": True, "full_map_type": 1, "few_shot": True},
]

# ====== Main Logic ======
rows = []

history_sizes = {
    "2x2": 5,
    "3x3": 5,
    "4x4": 10,
    "8x8": 15,
}
for config in configurations:
    print(f"\n== Running Configuration: {config['desc']} ==")

    h_size = history_sizes[map_name] if config["history"] else 0
    full_map_flag = config["full_map"]
    full_map_type = config["full_map_type"]
    fewshot_flag = config["few_shot"]

    # Tag for locating result folder
    h_tag = f"H{h_size}"
    o_tag = f"FM{full_map_type}" if full_map_flag else "PM"
    prompt_tag = f"{'Fewshot' if fewshot_flag else 'Zeroshot'}_basic_{h_tag}_{o_tag}"
    train_eval_tag = f"EvalEp{n_episodes_eval}EvalIntv{eval_interval}TrainEp{n_episodes}"
    env_id = f"{env_name}_map{map_name}_NotSlippery"
    experiment_dir = os.path.join(result_root, env_id, f"{llm_model_name}_{agent_name}", prompt_tag, train_eval_tag)

    # Run experiments for each seed
    for seed in seeds:
        cmd = [
            "python", toy_agent_script,
            "--update_type", "llm",
            "--seed", str(seed),
            "--map_name", map_name,
            "--max_new_tokens", str(max_new_tokens),
            "--history_window_size", str(h_size),
            "--n_episodes_eval", str(n_episodes_eval),
            "--n_episodes", str(n_episodes),
            "--eval_interval", str(eval_interval),
            "--llm_model_name", llm_model_name
        ]
        if full_map_flag:
            cmd += ["--full_map_observable", "--full_map_desc_type", str(full_map_type)]
        if fewshot_flag:
            cmd += ["--use_fewshot"]

        subprocess.run(cmd, check=True)

    # Collect results
    eval_returns = []
    for seed in seeds:
        result_file = os.path.join(experiment_dir, f"seed_{seed}", "results.json")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                data = json.load(f)
                eval = data.get("eval_ave_return")
                if isinstance(eval, (int, float)):
                    eval_returns.append(eval)
        else:
            print(f"[Warning] Missing result file: {result_file}")

    if eval_returns:
        mean = np.mean(eval_returns)
        sem = stats.sem(eval_returns)
    else:
        mean, sem = np.nan, np.nan
    print(f"[{config['desc']}] mean - {mean}, sem - {sem}")

    rows.append({
        "Configuration": config["desc"],
        "Mean": round(mean, 4) if not np.isnan(mean) else "NaN",
        "SEM": round(sem, 4) if not np.isnan(sem) else "NaN"
    })

    # ====== Save as CSV ======
    df = pd.DataFrame(rows)
    csv_name = f"map{map_name}_llm_{llm_model_name.replace('/', '-').replace(' ', '_')}.csv"
    df.to_csv(csv_name, index=False)
    print(f"\n✅ Results saved to: {csv_name}")

