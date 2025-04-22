import os
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import stats

def load_eval_returns(base_dir, seeds, is_rl=False):
    results = []
    for seed in seeds:
        result_path = os.path.join(base_dir, f"seed_{seed}", "results.json")
        with open(result_path, 'r') as f:
            data = json.load(f)
            eval_return = data["eval_ave_return"]
            if not is_rl:
                eval_return = float(eval_return)
            results.append(eval_return)
    return results

seeds = [1, 2, 3, 4, 5]

# Load data
maptype="map2x2" # map2x2 map4x4
surface_property="NotSlippery" #"IsSlippery" # NotSlippery
eval_episodes=25
train_episodes=5000 #15000
eval_interval=1 #1 #25 #100
eval_episode_points = list(np.arange(0, train_episodes+eval_interval, eval_interval))
eval_tag=f"{eval_episodes}EpisodesPerEvalEvery{eval_interval}Episodes_{train_episodes}TrainEpisodes"
plot_tag=f"{maptype}_{surface_property}_{eval_episodes}evalEpisodes_Every{eval_interval}Episodes"


root_dir="./" # this visualization script should be placed at the root of the llm4rl project folder
result_dir=f"{root_dir}/result/FrozenLake-v1_{maptype}_{surface_property}/TheBloke"
llm_returns = load_eval_returns(result_dir+f"/deepseek-llm-7b-chat-GPTQ-update_type_llm/{eval_tag}", seeds)
random_returns = load_eval_returns(result_dir+f"/deepseek-llm-7b-chat-GPTQ-update_type_random/{eval_tag}", seeds)
rl_returns = load_eval_returns(result_dir+f"/deepseek-llm-7b-chat-GPTQ-update_type_rl/{eval_tag}", seeds, is_rl=True)

# Compute stats
llm_mean = np.mean(llm_returns)
llm_sem = stats.sem(llm_returns)

random_mean = np.mean(random_returns)
random_sem = stats.sem(random_returns)

rl_array = np.array(rl_returns)  # shape (5, 20)
rl_mean = np.mean(rl_array, axis=0)
rl_sem = stats.sem(rl_array, axis=0)


# Plotting
plt.figure(figsize=(10, 6))

# Plot RL with x-axis as episodes
plt.plot(eval_episode_points, rl_mean, color='red', label='RL')
plt.fill_between(eval_episode_points, rl_mean - rl_sem, rl_mean + rl_sem, color='red', alpha=0.3)


# Plot LLM and Random as horizontal lines
## Set the expected x range for the horizontal lines 
xmin_val, xmax_val = 0, train_episodes
## Get the current x-axis limits
xmin, xmax = plt.xlim() 
## Calculate xmin and xmax as a fraction of the x axis
xmin_frac = (xmin_val - xmin) / (xmax - xmin)
xmax_frac = (xmax_val - xmin) / (xmax - xmin)

plt.axhline(llm_mean, xmin=xmin_frac, xmax=xmax_frac, color='blue', linestyle='-', label='LLM')
plt.fill_between(eval_episode_points, llm_mean - llm_sem, llm_mean + llm_sem, color='blue', alpha=0.2)

plt.axhline(random_mean, xmin=xmin_frac, xmax=xmax_frac, color='green', linestyle='-', label='Random')
plt.fill_between(eval_episode_points, random_mean - random_sem, random_mean + random_sem, color='green', alpha=0.2)

plt.xlabel("Episodes")
plt.ylabel("Average Return")
plt.title(f"Baseline Performance Comparison: {maptype} {surface_property}, {eval_episodes}EpisodesPerEval")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, f"baseline_performance_{plot_tag}.png"), dpi=300)
# Optional: save as PDF too
# plt.savefig(os.path.join(output_dir, "baseline_performance.pdf"))

#plt.show()

with open(os.path.join(output_dir, f"baseline_performance_{plot_tag}_summary.txt"), "a") as fm:
    print(f"Agent Type\tMean\tSEM\n")
    print(f"Random\t{random_mean}\t{random_sem}\n")
    print(f"LLM\t{llm_mean}\t{llm_sem}\n")
    print(f"RL\t{rl_mean.tolist()}\n\t{rl_sem.tolist()}\n")

    fm.write(f"Agent Type\tMean\tSEM\n")
    fm.write(f"Random\t{random_mean}\t{random_sem}\n")
    fm.write(f"LLM\t{llm_mean}\t{llm_sem}\n")
    fm.write(f"RL\t{rl_mean.tolist()}\n\t{rl_sem.tolist()}\n")


