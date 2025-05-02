from utility import collect_llm_experiences

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--llm_model_name', type=str, default="TheBloke/deepseek-llm-7b-chat-GPTQ")
parser.add_argument('--map_name', type=str, default="2x2")
parser.add_argument("--n_episodes_to_collect", type=int, default=100)

parser.add_argument('--full_map_observable', action='store_true', help='make the map fully observable to the agent.')
parser.add_argument('--full_map_desc_type', type=int, default=1, help='indicate the type of description of full map for the prompt. A higher value means a more explicit description of the map structure')
parser.add_argument('--use_chatgpt', action='store_true', help='Use a ChatGPT model as the llm module')
 
args = parser.parse_args()

llm_config = {
    "llm_model_name": args.llm_model_name,
    "history_window_size": 0,
    "full_map_observable": args.full_map_observable,
    "full_map_desc_type": 1,
    "generate_thought": False,
    "use_chatgpt": args.use_chatgpt,
    "max_new_tokens": 32,
    "use_fewshot": False,
    "debug": False
}

llm_model_short_name = llm_config['llm_model_name'].split('/')[-1]
history_tag = f"HWS{llm_config['history_window_size']}"
obs_tag = "PO"
if llm_config['full_map_observable']:
    obs_tag = f"FO{llm_config['full_map_desc_type']}"
chatgpt_tag = "ChatGPT" if llm_config['use_chatgpt'] else "NotChatGPT"
fewshot_tag = "UseFewshot" if llm_config['use_fewshot'] else "NoFewshot"
thought_tag = "HasThought" if llm_config['generate_thought'] else "NoThought"

map_name = args.map_name
n_episodes = args.n_episodes_to_collect #100, 1000
seeds=[1, 2, 3, 4, 5]
for seed in seeds:
    print(f"Collect trajectories using the LLM ({llm_config['llm_model_name']}) agent under the seed {seed}")
    save_path=f"./llm_agent_trajs/Map{map_name}_seed{seed}_{llm_model_short_name}_{history_tag}_{obs_tag}_{chatgpt_tag}_{fewshot_tag}_{thought_tag}_{n_episodes}Episodes.pkl"
    print(f"The collected trajectories will be saved to {save_path}")
    collect_llm_experiences(
        save_path=save_path,
        map_name=map_name,
        n_episodes=n_episodes,
        seed=seed,
        llm_config=llm_config
    )
