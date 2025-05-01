from utility import collect_llm_experiences


llm_config = {
    "llm_model_name": "TheBloke/deepseek-llm-7b-chat-GPTQ",
    "history_window_size": 5,
    "full_map_observable": False,
    "full_map_desc_type": 0,
    "generate_thought": False,
    "use_chatgpt": False,
    "max_new_tokens": 128,
    "use_fewshot": False,
    "debug": False
}

llm_model_short_name = llm_config['llm_model_name'].split('/')[-1]
history_tag = f"HWS{llm_config['history_window_size']}"
obs_tag = "PO"
if llm_config['full_map_observable']:
    obs_tag = "FO{llm_config['full_map_desc_type']}"
chatgpt_tag = "ChatGPT" if llm_config['use_chatgpt'] else "NotChatGPT"
fewshot_tag = "UseFewshot" if llm_config['use_fewshot'] else "NoFewshot"
thought_tag = "HasThought" if llm_config['generate_thought'] else "NoThought"

map_name = "3x3"
n_episodes = 100 #100, 1000
seed = 1 # 1, 2, 3, 4, 5
save_path=f"Map{map_name}_seed{seed}_{llm_model_short_name}_{history_tag}_{obs_tag}_{chatgpt_tag}_{fewshot_tag}_{thought_tag}_{n_episodes}Episodes"

collect_llm_experiences(
    save_path=f"{save_path}.pkl",
    map_name=map_name,
    n_episodes=n_episodes,
    seed=seed,
    llm_config=llm_config
)
