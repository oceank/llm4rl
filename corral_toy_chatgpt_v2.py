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

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    # This is the default and can be omitted
    api_key=openai_key,
)

def shaped_reward(observation):
    grid_size = 4
    current_row = observation // grid_size
    current_col = observation % grid_size
    obs = (current_row, current_col)
    goal_position = [3,3]
    # Reward shaping: Encourage moving closer to the goal
    distance_to_goal = np.linalg.norm(np.array(goal_position) - np.array(obs))

    return 1 - distance_to_goal/3/(2**0.5)

def print_map_desc(map_desc):
    for row in map_desc:
        print(f"\t{row}")
    print("")

class CorralFastIGW(object):
    def __init__(self, *, eta, nalgos, update_type, device):
        import numpy

        super(CorralFastIGW, self).__init__()
        self.eta = eta / nalgos
        self.invpalgo = torch.Tensor([ nalgos ] * nalgos).to(device)
        self.update_type = update_type

    def update(self, algo, invprop, reward):
        import numpy
        from scipy import optimize
        if self.update_type != 'corral':
            return
        assert torch.all(reward >= 0) and torch.all(reward <= 1), reward

        weightedlosses = self.eta * (-reward.squeeze(1)) * invprop.squeeze(1)

        newinvpalgo = torch.scatter(input=self.invpalgo,
                                    dim=0,
                                    index=algo,
                                    src=weightedlosses,
                                    reduce='add')

        invp = newinvpalgo.cpu().numpy()
        invp += 1 - numpy.min(invp)
        Zlb = 0
        Zub = 1
        while (numpy.sum(1 / (invp + Zub)) > 1):
            Zlb = Zub
            Zub *= 2
        root, res = optimize.brentq(lambda z: 1 - numpy.sum(1 / (invp + z)), Zlb, Zub, full_output=True)
        assert res.converged, res

        self.invpalgo = torch.tensor(invp + root, device=self.invpalgo.device)

    def sample_algo(self, N):
        initial_probs = 1.0/self.invpalgo
        if self.update_type == 'llm': #always use llm
          initial_probs[0] = 0
          initial_probs[1] = 1 # llm agent
        elif self.update_type == 'rl': #always use rl
          initial_probs[0] = 1 # rl agent
          initial_probs[1] = 0
        else: # corral
          initial_probs[0] = max(0.2, initial_probs[0])
          initial_probs[1] = 1 - initial_probs[0]

        algosampler = torch.distributions.categorical.Categorical(probs=initial_probs, validate_args=False)

        algo = algosampler.sample((N,))

        invpalgo = torch.gather(input=self.invpalgo.unsqueeze(0).expand(N, -1),
                                dim=1,
                                index=algo.unsqueeze(1))
        return algo, invpalgo


class llmModel:
    def __init__(self, model_name, map_config, is_slippery, full_map_observable=False, history_window=0, full_map_desc_type=0):
        self.is_slippery = is_slippery
        self.model_name = model_name
        self.map_config = map_config
        self.full_map_observable = full_map_observable
        self.map_desc_type = full_map_desc_type # 0, 1, 2
        self.map_text_desc = "" #"\t"+"\n\t".join(self.map_config) 
        if self.map_desc_type == 0:
            self.map_text_desc = "\t"+"\n\t".join(self.map_config) 
        else: #1
            for ri, row in enumerate(self.map_config):
                self.map_text_desc += f"\tRow {ri}: " + " ".join(row) + "\n"
        self.tile_meanings_text = "Tile meanings:\n- 'S' is the Start (you begin here)\n- 'F' is Frozen (safe to step on)\n- 'H' is a Hole (you fall and lose)\n- 'G' is the Goal (you win)\n"
        self.map_info_desc_for_prompt = "The whole map is not fully observable."


        if self.full_map_observable:
            if self.map_desc_type == 0:
               self.map_info_desc_for_prompt = f"The map is fully observable. The map is as follows:\n{self.map_text_desc}"
            else:  
                map_rows = len(self.map_config)
                self.map_info_desc_for_prompt = f"The map is fully observable. The map is a {map_rows}x{map_rows} grid as shown below (each letter represents a tile):\n{self.map_text_desc}\n{self.tile_meanings_text}"

        self.history_window = history_window
        self.history = []

        # Slippery property description
        self.slippery_property_desc = "The frozen surface is slippery. The intended action is executed with a 1/3 probability, while each of the perpendicular actions is also taken with a 1/3 probability."
        if not self.is_slippery:
            self.slippery_property_desc = "The frozen surface is not slippery such that the intended action is executed with a probability of 1."


    def format_history(self):
        """
        history: List of tuples [(position, surroundings, thought, action), ...]
        surroundings: dict with keys 'up', 'down', 'left', 'right'
        """
        lines = ["[History]: The following is a step-by-step history of the agent’s past decisions, listed in order from the first action to the most recent one."]

        for i, (pos, surroundings, thought, action) in enumerate(self.history, 1):
            surround_str = ", ".join(
                f"{dir_.capitalize()} = {val}" for dir_, val in surroundings.items()
            )
            lines.append(f"{i}. At {pos}, saw: {surround_str} →  Thought: {thought}, Action: {action}")
        return "\n".join(lines)


    def compose_prompt(self, observation):
        """Compose a detailed natural language prompt for DeepSeek with map context."""
        # the map configuration is not correct. Fix it by passing it when
        # initializing the LLM agent
        #map_config = [
        #    "SG",
        #    "HF"
        #]
        grid_size = len(self.map_config)

        # Find start and goal positions dynamically
        start_position = None
        goal_position = None
        for row in range(grid_size):
            for col in range(grid_size):
                if map_config[row][col] == "S":
                    start_position = (row, col)
                if map_config[row][col] == "G":
                    goal_position = (row, col)

        # Convert scalar observation to (row, col)
        current_row = observation // grid_size
        current_col = observation % grid_size

        # Define surroundings
        surroundings = {
            "up": map_config[current_row - 1][current_col] if current_row > 0 else "Edge",
            "down": map_config[current_row + 1][current_col] if current_row < grid_size - 1 else "Edge",
            "left": map_config[current_row][current_col - 1] if current_col > 0 else "Edge",
            "right": map_config[current_row][current_col + 1] if current_col < grid_size - 1 else "Edge"
        }

        tile_descriptions = {
            "S": "Start",
            "F": "Frozen",
            "H": "Hole",
            "G": "Goal",
            "Edge": "Edge of the map"
        }
        surroundings_text = {direction: tile_descriptions[tile] for direction, tile in surroundings.items()}

        # Create the prompt
        # - The currently provided information is more-like a partial observation
        # of the agent with a view of 1-step size.
        system_content = (
            f"You are playing Frozen Lake, a grid-based game. The objective is to move from the start tile {start_position} (marked as S) "
            f"to the goal tile {goal_position} (marked as G) without falling into holes (H) on the grid. {self.slippery_property_desc}\n"
            "At each step, you can take one of the following actions:\n"
            "- 'left': Move left\n"
            "- 'right': Move right\n"
            "- 'up': Move up\n"
            "- 'down': Move down\n\n"
            "Be careful to avoid holes and aim to reach the goal as quickly as possible.\n"
            f"{self.map_info_desc_for_prompt}"
        )

        #user_content
        current_history = ""
        if len(self.history) != 0:
            current_history =  self.format_history()
            
        current_query = (
            f"[Current Query]\n"
            f"You are currently at position ({current_row}, {current_col}), which is a {tile_descriptions[map_config[current_row][current_col]]} tile.\n"
            f"Surrounding tiles:\n"
            f"- Up: {surroundings_text['up']}\n"
            f"- Down: {surroundings_text['down']}\n"
            f"- Left: {surroundings_text['left']}\n"
            f"- Right: {surroundings_text['right']}\n\n"
            "What should you do next? Please respond with one of: 'move left', 'move right', 'move up', 'move down'."
        )
        
        if current_history != "":
            user_content = f"{current_history}\n\n{current_query}"
        else:
            user_content = current_query
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        if self.history_window != 0:
            surrounding_desc = {k: tile_descriptions[v] for k, v in surroundings.items()}
            self.history.append(((current_row, current_col), surrounding_desc, None, None))

        #llm_msg = "llm_msg:"
        #for msg in messages:
        #    llm_msg += f"\n\t[{msg['role']}] {msg['content']}\n"
        #print(llm_msg)

        #return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        return messages

    def extract_action_from_response(self, response):
        valid_actions = ['move left', 'move right', 'move up', 'move down']
        response_lower = response.lower()
        is_random_action = False

        for action in valid_actions:
            if action in response_lower:
                return action, is_random_action

        is_random_action = True
        return random.choice(valid_actions), is_random_action

    def generate_action(self, observation, debug=False):
        prompt = self.compose_prompt(observation)
        try:
            # action_text is actually is the thought of the LLM
            action_text = client.chat.completions.create(
                messages=prompt,
                model=self.model_name, #"gpt-4o-mini",
            ).choices[0].message.content.strip()
           
            chosen_action, is_random_action = self.extract_action_from_response(action_text)
            if self.history_window != 0:
                pos, surrounding_desc, _, _ = self.history.pop()
                self.history.append((pos, surrounding_desc, action_text, chosen_action))
                if len(self.history) > self.history_window:
                    self.history.pop(0)

            if debug:
                msg = f"[LLM Thought]: {action_text}" + f"\n\n[Next Action Text]: {chosen_action}. (Random Action: {is_random_action})"
                print(msg)

            return chosen_action
        
        except Exception as e:
            print(f"An error occurred: {e}")
            print('invalid observation: ',observation)
            fallback_action = random.choice(["left", "right", "up", "down"])
            return fallback_action  # Ensure a valid action is always returned
        

    def match_action(self, text_action):
        if text_action == "move left":
            return 0
        elif text_action == "move down":
            return 1
        elif text_action == "move right":
            return 2
        elif text_action == "move up":
            return 3
        else:
            return random.choice([0, 1, 2, 3])



class FrozenLakeAgent:
    def __init__(self, learning_rate, final_learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor, update_type, map_config, is_slippery, full_map_observable, llm_model_name, history_window, full_map_desc_type):
        self.q_values = defaultdict(lambda: np.zeros(4))
        self.lr = learning_rate
        self.final_lr = final_learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.update_type = update_type
        self.corral_sampler = CorralFastIGW(eta=0.05, nalgos=2, update_type=self.update_type, device=self.device)
        self.corral_list = []
        self.llm_model = llmModel(llm_model_name, map_config, is_slippery, full_map_observable, history_window, full_map_desc_type)
        self.llm_model_name = llm_model_name
        self.is_slippery = is_slippery

    def get_action(self, obs, use_llm):
        if use_llm:
            text_action = self.llm_model.generate_action(obs)
            return self.llm_model.match_action(text_action)
        else:
            if np.random.random() < self.epsilon: # epsilon-exploration
                return env.action_space.sample()
            else: # action derived from the max q value
                return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q_value
        lr = self.get_lr()
        self.q_values[obs][action] += lr * (target - self.q_values[obs][action])
        self.training_error.append(target - self.q_values[obs][action])
        return target

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def get_lr(self):
        if self.epsilon == self.final_epsilon: # generally, self.final_epsilon is 0 or close to 0
            return self.final_lr
        else:
            return self.lr

#def evaluate_agent(agent, env_name, map_name, map_config, is_slippery, n_episodes_eval=10):
def evaluate_agent(agent, eval_env, n_episodes_eval=10, debug=False):
    #eval_env = gym.make(env_name, is_slippery=is_slippery, map_name=map_name, desc=map_config)
    eval_returns = []

    for episode in tqdm(range(n_episodes_eval)):
        obs, _ = eval_env.reset()
        print(f"[Map Info]:")
        print_map_desc(eval_env.spec.kwargs['desc'])

        done = False
        episode_return = 0
        step = 0

        while not done:
            # Greedy policy: choose the best Q action if RL, or use LLM
            if agent.update_type == "random":
                action = eval_env.action_space.sample()
            elif agent.update_type == "llm":
                if debug:
                    print(f"\n===> Episode {episode}, Step {step}")
                    step += 1
                text_action = agent.llm_model.generate_action(obs, debug=debug)
                action = agent.llm_model.match_action(text_action)
            else: # "rl"
                q_values = agent.q_values[obs]
                action = int(np.argmax(q_values))

            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_return += reward
            done = terminated or truncated
        if debug:
            print(f"\n===> Episode {episode} finishes.\n")

        eval_returns.append(episode_return)

    avg_return = np.mean(eval_returns)
    return avg_return, eval_returns


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--discount_factor", type=float, default=0.9)
    parser.add_argument('--learning_rate', type=float, default=0.9)
    parser.add_argument('--final_learning_rate', type=float, default=0.0001)
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
    parser.add_argument('--full_map_desc_type', type=int, default=0, help='the type of map description')

    args = parser.parse_args()



    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #epsilon_decay = args.start_epsilon / (args.n_episodes / 2)

    llm_model_name = "gpt-4o-2024-08-06" #"TheBloke/deepseek-llm-7b-chat-GPTQ" #"deepseek-ai/deepseek-llm-7b-chat"
    map_configs = {
            "2x2OneStep": [
                "SG",
                "HF",
                ],
            "2x2": [
                "SF",
                "HG",
                ],
            "3x3":[
                "SFF",
                "FFH",
                "HFG"
                ],
            "4x4":[
                "SFFF",
                "FHFH",
                "FFFH",
                "HFFG"
                ],
            "8x8": [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFFHFFFF",
                "FFFFFHFF",
                "FFFHFFFF",
                "FHHFFFHF",
                "FHFFHFHF",
                "FFFHFFFG",
            ]
    }
    map_name = args.map_name  #'4x4' #'2x2'
    map_config = map_configs[map_name]
    env_name = 'FrozenLake-v1'
    is_slippery=args.is_slippery
    env_id = f"{env_name}_map{map_name}_" + ("IsSlippery" if is_slippery else "NotSlippery")

    history_window = 0 #5
    agent = FrozenLakeAgent(
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        initial_epsilon=args.start_epsilon,
        epsilon_decay=args.epsilon_decay,
        final_epsilon=args.final_epsilon,
        discount_factor=args.discount_factor,
        update_type=args.update_type,
        map_config = map_config,
        is_slippery = is_slippery,
        full_map_observable = args.full_map_observable,
        llm_model_name = llm_model_name,
        history_window = history_window,
        full_map_desc_type = args.full_map_desc_type
    )

    env = gym.make(env_name, is_slippery=is_slippery, map_name=map_name, desc=map_config)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=args.n_episodes)
    obs, info = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    eval_env = gym.make(env_name, is_slippery=is_slippery, map_name=map_name, desc=map_config)
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env, deque_size=args.n_episodes)
    obs, info = eval_env.reset(seed=args.seed+1000000)
    eval_env.action_space.seed(args.seed+1000000)


    returns = []
    debug=True
    # evaluating the baseline llm agent
    if args.update_type == "llm" or args.update_type == "random":
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
    update_type_info = f"update_type_{args.update_type}" + ("" if args.update_type=="random" else ("_FullMapInfo" if args.full_map_observable else "_PartialMapInfo"))
    print(update_type_info)
    save_dir = f'./result/{env_id}/{llm_model_name}-{update_type_info}/{args.n_episodes_eval}EpisodesPerEvalEvery{args.eval_interval}Episodes_{args.n_episodes}TrainEpisodes/seed_{args.seed}/'
    os.makedirs(save_dir, exist_ok=True)

    with open(save_dir + 'results.json', 'w') as fp:
        json.dump(results, fp)

