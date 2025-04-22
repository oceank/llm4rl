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

from dotenv import load_dotenv

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
          initial_probs[1] = 1
        elif self.update_type == 'rl': #always use rl
          initial_probs[0] = 1
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
    def __init__(self, map_config, random_probs=None, is_slippery=False):
        dtype = torch.bfloat16
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #model_name = "mistralai/Mistral-Nemo-Instruct-2407"

        #self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #self.model = AutoModelForCausalLM.from_pretrained(
        #    model_name, torch_dtype=dtype, device_map="auto"
        #).to(self.device)

        #self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        #self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

        self.map_config = map_config
        self.history = []  # Store visited positions and states

    def compose_prompt(self, observation):
        grid_size = len(self.map_config)

        # Convert scalar observation to (row, col)
        current_row = observation // grid_size
        current_col = observation % grid_size
        current_position = (current_row, current_col)

        # Define surroundings (only based on visited tiles)
        tile_descriptions = {
            "S": "Start position",
            "F": "Frozen (Safe to walk on)",
            "H": "Hole (Dangerous, leads to failure)",
            "G": "Goal (Win the game)",
            "Edge": "Wall (Cannot move here)"
        }

        surroundings = {}
        for direction, (d_row, d_col) in [("up", (-1, 0)), ("down", (1, 0)), ("left", (0, -1)), ("right", (0, 1))]:
            new_row, new_col = current_row + d_row, current_col + d_col
            if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
                surroundings[direction] = tile_descriptions[self.map_config[new_row][new_col]]
            else:
                surroundings[direction] = tile_descriptions["Edge"]

        # Store observation history (visited tiles)
        self.history.append({
            "position": current_position,
            "tile": tile_descriptions[self.map_config[current_row][current_col]],
            "surroundings": surroundings,
            "action": None  # Will be updated later after taking action
        })
        if len(self.history) > 10:  # Keep the last 5 observations
            self.history.pop(0)

        history_text = "\n".join([
            f"Step {i+1}: Position {h['position']} | Surroundings: {h['surroundings']}"
            for i, h in enumerate(self.history)
        ])

        # Partial Observability Strategy
        visited_holes = [h['position'] for h in self.history if h['tile'] == "Hole (Dangerous, leads to failure)"]
        visited_goal = [h['position'] for h in self.history if h['tile'] == "Goal (Win the game)"]

        hole_info = f"Known Holes (visited before): {visited_holes}" if visited_holes else "No known holes yet."
        goal_info = f"Goal reached before at {visited_goal}, navigate accordingly." if visited_goal else "Goal position unknown."

        optimal_path_heuristic = (
            "### **Navigation Strategy:**\n"
            "- Move strategically to explore new tiles while avoiding known holes. **Holes end the game immediately.**\n"
            "- **If moving forward would trap you between holes, backtrack early before reaching a dead-end.**\n"
            "- **A dead-end is a position where all forward moves lead to holes. If you reach one, immediately return to a safe tile.**\n"
            "- **If you have already backtracked from a dead-end once, do NOT go back down that path again.**\n"
            "- Keep track of visited safe paths and avoid backtracking unnecessarily, **but backtracking is always better than a dead-end.**\n"
        )


        system_content = (
            "You are playing Frozen Lake, a grid-based game with **partial observability**.\n\n"
            "### **Game Rules:**\n"
            "- Reach the goal without falling into holes.\n"
            "- You only know about tiles you have previously visited.\n"
            "- Movement is **deterministic**, meaning you always move in the chosen direction.\n\n"
            "### **Available Actions:**\n"
            "- 'left': Move left\n"
            "- 'right': Move right\n"
            "- 'up': Move up\n"
            "- 'down': Move down\n\n"
            + optimal_path_heuristic
        )

        user_content = (
            f"### **Recent Observations:**\n"
            f"{history_text}\n\n"
            "**What action should you take next?** Respond with ONLY one of: 'left', 'right', 'up', 'down'."
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        #print('-'*100)
        #print(messages)
        #return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        return messages

    def reset_history(self):
        self.history = []

    def extract_action_from_response(self, response):
        #print('response: ', response)
        valid_actions = ['left', 'right', 'up', 'down']
        response_lower = response.lower()
        
        for action in valid_actions:
            if action in response_lower:
                return action
        return random.choice(valid_actions)

    def generate_action(self, observation):
        prompt = self.compose_prompt(observation)
        #input_tensor = prompt.to(self.device)

        #with torch.no_grad():
        #    outputs = self.model.generate(input_tensor, max_new_tokens=50)

        #new_tokens = outputs[:, input_tensor.shape[1]:]
        #action_text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True).lower()
        try:
            action_text = client.chat.completions.create(
                messages=prompt,
                model="gpt-4o-mini",
            ).choices[0].message.content.strip()
            
            chosen_action = self.extract_action_from_response(action_text)
            self.history[-1]["action"] = chosen_action
            
            return chosen_action
        
        except:
            print('invalid observation: ',observation)
            fallback_action = random.choice(["left", "right", "up", "down"])
            return fallback_action  # Ensure a valid action is always returned
        

    def match_action(self, text_action):
        return {"left": 0, "down": 1, "right": 2, "up": 3}.get(text_action, random.choice([0, 1, 2, 3]))

class FrozenLakeAgent:
    def __init__(self, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor, update_type, map_config, random_probs, is_slippery):
        self.q_values = defaultdict(lambda: np.zeros(4))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.update_type = update_type
        self.corral_sampler = CorralFastIGW(eta=0.05, nalgos=2, update_type=self.update_type, device=self.device)
        self.corral_list = []
        self.llm_model = llmModel(map_config, random_probs, is_slippery)

    def get_action(self, obs, use_llm):
        if use_llm:
            text_action = self.llm_model.generate_action(obs)
            return self.llm_model.match_action(text_action)
        elif np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q_value
        self.q_values[obs][action] += self.lr * (target - self.q_values[obs][action])
        self.training_error.append(target - self.q_values[obs][action])
        return target

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--n_episodes', type=int, default=100000)
    parser.add_argument('--start_epsilon', type=float, default=1.0)
    parser.add_argument('--final_epsilon', type=float, default=0.1)
    parser.add_argument('--random_probs', type=float, default=0.1)
    parser.add_argument('--is_slippery', type=int, default=0)
    parser.add_argument('--update_type', type=str, default='rl', help="Supports [rl, llm, corral]")
    args = parser.parse_args()
    
    is_slippery = bool(args.is_slippery)
    print('is_slippery: ',is_slippery)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    epsilon_decay = args.start_epsilon / (args.n_episodes / 2)
    
    save_dir = f'./results/random_probs_{args.random_probs}/is_slippery_{str(is_slippery)}/GPT4o-mini/update_type_{args.update_type}/seed_{args.seed}/'
        
    '''
    map_config = [
        "SG",
        "HF"
    ]
    '''
    map_config = [
        "SFFF",
        "FHFH",
        "FHFH",
        "HFFG"
        ]

    
    agent = FrozenLakeAgent(
        learning_rate=args.learning_rate,
        initial_epsilon=args.start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=args.final_epsilon,
        discount_factor=0.95,
        update_type=args.update_type,
        map_config = map_config,
        random_probs= args.random_probs,
        is_slippery = is_slippery
    )

    #env = gym.make('FrozenLake-v1', is_slippery=False, map_name="2x2", desc=map_config)
    env = gym.make('FrozenLake-v1', is_slippery=is_slippery, map_name="4x4", desc=map_config, random_prob = args.random_probs)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=args.n_episodes)
    
    returns = []
    for episode in tqdm(range(args.n_episodes)):
        obs, info = env.reset()
        agent.llm_model.reset_history()
        done = False
        #algo, invpalgo = agent.corral_sampler.sample_algo(1)
        #agent.corral_list.append(algo[0].item())
        return_i = []
        corral_list_i = []
        steps = 0
        while not done:
            steps += 1
            algo, invpalgo = agent.corral_sampler.sample_algo(1)
            corral_list_i.append(algo[0].item())
            action = agent.get_action(obs, bool(algo[0].item()))
            next_obs, reward, terminated, truncated, info = env.step(action)
            #reward = shaped_reward(next_obs)
            if reward == 1:
                reward = reward/steps
            return_i.append(reward)
            target = agent.update(obs, action, reward, terminated, next_obs)
            done = terminated or truncated
            obs = next_obs
            
            #normalized_target = np.clip(target/(1+agent.discount_factor), 0, 1)
            normalized_target = np.clip(target*(1-agent.discount_factor), 0, 1)
            corral_reward = torch.tensor([normalized_target]).float().view(1, 1).to(agent.device)
            agent.corral_sampler.update(algo, invpalgo, corral_reward)
        #print('return_i: ',return_i)
        agent.corral_list.append(corral_list_i)
        returns.append(np.sum(return_i))
        #returns.append(return_i)
        #corral_reward = torch.tensor([reward]).float().view(1, 1).to(agent.device)
        #agent.corral_sampler.update(algo, invpalgo, corral_reward)
        agent.decay_epsilon()

    os.makedirs(save_dir, exist_ok=True)
    results = {
        'return': returns,
        'length': [float(x) for x in np.array(env.length_queue).flatten()],
        'training_error': [float(x) for x in np.array(agent.training_error).flatten()],
        'corral_list': agent.corral_list
    }
    print('-'*105)
    print('returns: ',np.mean(returns))

    with open(save_dir + 'results.json', 'w') as fp:
        json.dump(results, fp)

