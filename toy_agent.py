from __future__ import annotations
import torch
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from openai import OpenAI
import argparse
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import random
import scipy.stats as stats
from utility import shaped_reward, llmModel, CorralFastIGW, FrozenLakeAgent, evaluate_agent, map_configs 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--discount_factor", type=float, default=0.9)
    parser.add_argument('--learning_rate', type=float, default=0.9)
    parser.add_argument('--final_learning_rate', type=float, default=0.0001)
    parser.add_argument('--train_budget', type=int, default=100000)
    parser.add_argument('--n_episodes', type=int, default=100000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--n_episodes_eval', type=int, default=10)
    parser.add_argument('--start_epsilon', type=float, default=1.0)
    parser.add_argument('--final_epsilon', type=float, default=0.0)
    parser.add_argument('--epsilon_decay', type=float, default=0.0001)
    parser.add_argument('--update_type', type=str, default='rl', help="Supports [rl, llm, corral]")
    parser.add_argument('--map_name', type=str, default='2x2', help="the name of the map used in FrozenLake")
    parser.add_argument('--is_slippery', action='store_true', help='enable the slippery perperty of the frozen surface.')
    parser.add_argument('--full_map_observable', action='store_true', help='make the map fully observable to the agent.')
    parser.add_argument('--full_map_desc_type', type=int, default=0, help='indicate the type of description of full map for the prompt. A higher value means a more explicit description of the map structure')
    parser.add_argument('--generate_thought', action='store_true', help='make the LLM agent first generate its thought on the current situation and then combine its thought and the current sitation to make a decision')
    parser.add_argument('--history_window_size', type=int, default=0, help='the maximum number of the past experiences that can be incorporated into the query to the LLM. A value of 0 indicate no history is used. A common non-zero value is 5 for the current experiment.')
    parser.add_argument('--llm_model_name', type=str, default="TheBloke/deepseek-llm-7b-chat-GPTQ", help="The name of the LLM model")
    parser.add_argument('--use_chatgpt', action='store_true', help='Use a ChatGPT model as the llm module')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='the max number of tokens for newly generated text')
    parser.add_argument('--use_fewshot', action='store_true', help='Use few-shot examples')
 
    args = parser.parse_args()

    chatgpt_client=None
    if args.use_chatgpt:
        from dotenv import load_dotenv

        load_dotenv()
        openai_key = os.getenv("OPENAI_API_KEY")

        chatgpt_client = OpenAI(
            # This is the default and can be omitted
            api_key=openai_key,
        )

    debug=True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #epsilon_decay = args.start_epsilon / (args.n_episodes / 2)

    milestones = [] #[(1, 1), (2, 2)], currently only use it with the map 3x3
    agent_type_short_names = {
        "random": "RND",
        "llm"   : "LLM",
        "rl"    : "RL",
    }
    llm_model_name = args.llm_model_name #"TheBloke/deepseek-llm-7b-chat-GPTQ" #"deepseek-ai/deepseek-llm-7b-chat"

    map_name = args.map_name  #'4x4' #'2x2'
    map_config = map_configs[map_name]
    env_name = 'FrozenLake-v1'

    # naming format of the result folder: env_map/llm/prompt/train_eval/seed 
    agent_name = agent_type_short_names[args.update_type]
    env_id = f"{env_name}_map{map_name}_" + ("IsSlippery" if args.is_slippery else "NotSlippery")
    
    h_tag = f"H{args.history_window_size}"
    o_tag = f"FM{args.full_map_desc_type}" if args.full_map_observable else "PM"
    prompt_tag = f"basic_{h_tag}_{o_tag}"
    if len(milestones) > 0:
        prompt_tag += "_UseMilestones"
    if args.use_fewshot:
        prompt_tag += "_Fewshot"
    else:
        prompt_tag += "_Zeroshot"
    train_eval_tag = f"EvalEp{args.n_episodes_eval}EvalIntv{args.eval_interval}TrainEp{args.n_episodes}"
    save_dir = f'./result/{env_id}/{llm_model_name}_{agent_name}/{prompt_tag}/{train_eval_tag}/seed_{args.seed}/'
    os.makedirs(save_dir, exist_ok=True)


    # initialize the agent
    agent = FrozenLakeAgent(
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        initial_epsilon=args.start_epsilon,
        epsilon_decay=args.epsilon_decay,
        final_epsilon=args.final_epsilon,
        discount_factor=args.discount_factor,
        update_type=args.update_type,
        map_config = map_config,
        is_slippery = args.is_slippery,
        llm_model_name = llm_model_name,
        history_window_size = args.history_window_size,
        milestones = milestones,
        full_map_observable = args.full_map_observable,
        full_map_desc_type = args.full_map_desc_type,
        generate_thought = args.generate_thought,
        debug = debug,
        use_chatgpt = args.use_chatgpt,
        chatgpt_client = chatgpt_client,
        max_new_tokens = args.max_new_tokens,
        use_fewshot = args.use_fewshot,
    )

    # initialize the training and evaluation envs
    env = gym.make(env_name, is_slippery=args.is_slippery, map_name=map_name, desc=map_config)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=args.n_episodes)
    obs, info = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    eval_env = gym.make(env_name, is_slippery=args.is_slippery, map_name=map_name, desc=map_config)
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env, deque_size=args.n_episodes)
    obs, info = eval_env.reset(seed=args.seed+1000000)
    eval_env.action_space.seed(args.seed+1000000)


    # training and/or evaluating the agent
    returns = []
    if args.update_type == "llm" or args.update_type == "random":
        # evaluating the baseline llm agent
        eval_ave_return, eval_returns = evaluate_agent(agent, eval_env, n_episodes_eval=args.n_episodes_eval, debug=debug)
        results = {
            'eval_ave_return': eval_ave_return,
        }
        print('-'*105)
        print(f'evaluation returns: {eval_ave_return} ({eval_returns})')

    else: # "rl" or "corral"
        returns = []
        eval_returns = []

        episode = 0
        eval_ave_return, eval_returns_per_evaluation = evaluate_agent(agent, eval_env, n_episodes_eval=args.n_episodes_eval)
        eval_returns.append(eval_ave_return)
        print(f'[Episode {episode}] evaluation returns: {eval_ave_return} ({eval_returns})')

        # training
        for episode in tqdm(range(1, args.n_episodes+1, 1)):
            obs, info = env.reset()
            done = False
            #algo, invpalgo = agent.corral_sampler.sample_algo(1)
            #agent.corral_list.append(algo[0].item())
            return_i = []
            corral_list_i = []
            while not done:
                algo, invpalgo = agent.corral_sampler.sample_algo(1)
                corral_list_i.append(algo[0].item())
                action = agent.get_action(obs, bool(algo[0].item()))
                next_obs, reward, terminated, truncated, info = env.step(action)
                #reward = shaped_reward(next_obs)
                return_i.append(reward)
                target = agent.update(obs, action, reward, terminated, next_obs)
                done = terminated or truncated
                obs = next_obs

                normalized_target = np.clip(target/(1+agent.discount_factor), 0, 1)
                corral_reward = torch.tensor([normalized_target]).float().view(1, 1).to(agent.device)
                agent.corral_sampler.update(algo, invpalgo, corral_reward)

            agent.corral_list.append(corral_list_i)
            #returns.append(np.mean(return_i))
            returns.append(np.sum(return_i))
            #corral_reward = torch.tensor([reward]).float().view(1, 1).to(agent.device)
            #agent.corral_sampler.update(algo, invpalgo, corral_reward)
            agent.decay_epsilon()

            if episode%args.eval_interval == 0:
                eval_ave_return, eval_returns_per_evaluation = evaluate_agent(agent, eval_env, n_episodes_eval=args.n_episodes_eval)
                eval_returns.append(eval_ave_return)
                print(f'[Episode {episode}] evaluation returns: {eval_ave_return} ({eval_returns})')


        results = {
            'eval_ave_return': eval_returns,
            'training_return': returns,
            'training_length': [float(x) for x in np.array(env.length_queue).flatten()],
            'training_error': [float(x) for x in np.array(agent.training_error).flatten()],
            'training_corral_list': agent.corral_list
            }
 
    with open(save_dir + 'results.json', 'w') as fp:
        json.dump(results, fp)

