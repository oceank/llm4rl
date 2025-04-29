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
    def __init__(self, model_name, map_config, is_slippery, full_map_observable=False, history_window_size=0, milestones=[], full_map_desc_type=0, generate_thought=False, debug=False):
        self.debug = debug
        self.is_slippery = is_slippery
        self.model_name = model_name
        self.map_config = map_config
        self.generate_thought = generate_thought
        self.full_map_observable = full_map_observable
        self.full_map_desc_type = full_map_desc_type

        self.tile_descriptions = {
            "S": "Start",
            "F": "Frozen",
            "H": "Hole",
            "G": "Goal",
            "E": "Edge of the map"
        }

        self.get_start_and_goal_position()

        self.map_info_desc_for_prompt = "The whole map is not fully observable."
        self.tile_meanings_text = "Tile meanings:\n- 'S' is the Start (you begin here)\n- 'F' is Frozen (safe to step on)\n- 'H' is a Hole (you fall and lose)\n- 'G' is the Goal (you win)\n"

        if self.full_map_observable:
            self.get_map_desc(full_map_desc_type=self.full_map_desc_type)


        self.history_window_size = history_window_size
        self.history = []

        self.milestones = milestones # list of tuples, each represents the location of the milestone
        self.current_milestone = None
        self.current_milestone_idx = -1
        if len(self.milestones) > 0:
            self.curent_milestone_idx = 0
            self.current_milestone = self.milestones[self.curent_milestone_idx]
            self.current_goal_position = self.current_milestone

        self.dtype=torch.float16 #torch.bfloat16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # model_name = "deepseek-ai/deepseek-llm-7b-chat"

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=self.dtype, device_map="auto", trust_remote_code=True, revision="main"
        ).to(self.device)

        # Put PyTorch model in eval mode if applicable
        if hasattr(self.model, "eval"):
            self.model.eval()

        # Set generation configuration
        self.model.generation_config = GenerationConfig.from_pretrained(self.model_name)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

        # Slippery property description
        self.slippery_property_desc = "The frozen surface is slippery. The intended action is executed with a 1/3 probability, while each of the perpendicular actions is also taken with a 1/3 probability."
        if not self.is_slippery:
            self.slippery_property_desc = "The frozen surface is not slippery such that the intended action is executed with a probability of 1."

    def get_start_and_goal_position(self):
        self.grid_size = len(self.map_config)
        # Find start and goal positions dynamically
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if map_config[row][col] == "S":
                    self.start_position = (row, col)
                if map_config[row][col] == "G":
                    self.goal_position = (row, col)

        self.current_start_position = self.start_position
        self.current_goal_position = self.goal_position

    def get_current_position(self, observation):
        current_row = observation // self.grid_size
        current_col = observation % self.grid_size
        current_position = (current_row, current_col)
        return current_position

    def get_map_desc(self, full_map_desc_type=0):
        self.map_text_desc = ""
        if full_map_desc_type == 0:
            self.map_text_desc = "\t"+"\n\t".join(self.map_config)
        else:
            if full_map_desc_type == 1:
                for ri, row in enumerate(self.map_config):
                    self.map_text_desc += f"\tRow {ri}: " + " ".join(row) + "\n"
            else: # a more explicit description
                for ri, row in enumerate(self.map_config):
                    for ci, v in enumerate(row):
                        self.map_text_desc += f"\t Row {ri}, Column {ci}: the tile is {v}.\n"

            map_rows = len(self.map_config)
            self.map_info_desc_for_prompt = f"The map is fully observable. The map is a {map_rows}x{map_rows} grid as shown below (each letter represents a tile):\n{self.map_text_desc}\n{self.tile_meanings_text}"


    def get_system_content_for_prompt(self):
        # used parameters: self.current_start_position, self.current_goal_position, self.slippery_property_desc and self.map_info_desc_for_prompt
        system_content = (
            f"You are playing Frozen Lake, a grid-based game. The objective is to move from the start tile {self.current_start_position} (marked as S) "
            f"to the goal tile {self.current_goal_position} (marked as G) without falling into holes (H) on the grid. {self.slippery_property_desc}\n"
            "At each step, you can take one of the following actions:\n"
            "- 'left': Move left\n"
            "- 'right': Move right\n"
            "- 'up': Move up\n"
            "- 'down': Move down\n\n"
            "Be careful to avoid holes and aim to reach the goal as quickly as possible.\n"
            f"{self.map_info_desc_for_prompt}"
        )
        return system_content

    def get_surrounding_text_desc(self, current_position):
        current_row, current_col = current_position
        # Define surroundings: 'E' indicates an edge of the map
        surroundings = {
            "Up"   : self.map_config[current_row - 1][current_col] if current_row > 0 else "E",
            "Down" : self.map_config[current_row + 1][current_col] if current_row < self.grid_size - 1 else "E",
            "Left" : self.map_config[current_row][current_col - 1] if current_col > 0 else "E",
            "Right": self.map_config[current_row][current_col + 1] if current_col < self.grid_size - 1 else "E",
        }

        surroundings_text = {direction: self.tile_descriptions[tile] for direction, tile in surroundings.items()}
        return surroundings_text


    def get_current_query(self, current_position, surroundings_text_dict, generate_thought = False, thought = None):
        current_row, current_col = current_position
        context = (
            f"You are currently at position {current_position}, which is a {self.tile_descriptions[self.map_config[current_row][current_col]]} tile.\n"
            f"Surrounding tiles:\n"
            f"- Up: {surroundings_text_dict['Up']}\n"
            f"- Down: {surroundings_text_dict['Down']}\n"
            f"- Left: {surroundings_text_dict['Left']}\n"
            f"- Right: {surroundings_text_dict['Right']}\n\n"
        )
        if generate_thought:
            thought_query = "What should you do next? Before answering, analyze the situation step by step and think alound as you make the decision."
            current_query = context + thought_query
        else:
            decision_query = "What should you do next? Please respond with one of actions: 'move left', 'move right', 'move up', 'move down'."
            if thought is not None:
                decision_query = f"Your analysis of the current situation: {thought}\n\n" + decision_query
            current_query = context + decision_query
        return current_query

    def format_history(self):
        """
        history: List of tuples [(position, surroundings, thought, action), ...]
        surroundings: dict with keys 'move up', 'move down', 'move left', 'move right'
        """
        lines = ["[History]: The following is a step-by-step history of the agent’s past decisions, listed in order from the first action to the most recent one."]

        for i, (pos, surroundings, thought, action) in enumerate(self.history, 1):
            surround_str = ", ".join(
                f"{dir_.capitalize()} = {val}" for dir_, val in surroundings.items()
            )
            lines.append(f"{i}. At {pos}, saw: {surround_str} →  Thought: {thought}, Action: {action}")
        return "\n".join(lines)


    def compose_prompt_basic(self, observation, generate_thought=False, thought=None):
        # Convert scalar observation to (row, col)
        current_position = self.get_current_position(observation)

        # When no milestones are provided, self.current_milestone is None, so the IF branch will be skipped.
        # When milestones are provided and the current milestone is reached, then update it.
        #   When the final goal is reached, the agent will stop running the LLM for the current task.
        #   That is, this part of codes will not be reached then.
        if self.current_milestone == current_position:
            self.curent_milestone_idx += 1
            self.current_milestone = self.milestones[self.curent_milestone_idx]
            self.current_start_position = current_position
            self.current_goal_position = self.current_milestone

        # Define surroundings
        surroundings_text_dict = self.get_surrounding_text_desc(current_position)

        # Create the prompt
        # - The currently provided information is more-like a partial observation
        # of the agent with a view of 1-step size.
        system_content = self.get_system_content_for_prompt()
        current_query = self.get_current_query(current_position, surroundings_text_dict, generate_thought, thought)

        # compose the description of the history if it is enabled
        user_content = current_query
        if self.history_window_size != 0: # incorpate the history into the prompt
            # prepend the history description to the current_query
            if len(self.history) != 0:
                current_history =  self.format_history()
                user_content = f"{current_history}\n\n{current_query}"
            # when having the decision query, initialize the current LLM experience and add it into the history.
            # the generated thought will be added into history when processing the decision query, so skip the history update here.
            if not generate_thought:
                self.history.append((current_position, surroundings_text_dict, thought, None)) # current position, surroundings, thought, action

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        # debug the prompt
        if False: #self.debug:
            llm_msg = "[Debug] llm_msg:"
            for msg in messages:
                llm_msg += f"\n\t[{msg['role']}] {msg['content']}\n"
            print(llm_msg)

        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    def compose_thought_prompt(self, observation):
        return self.compose_prompt_basic(observation, generate_thought=True)

    def compose_prompt(self, observation, thought=None):
        return self.compose_prompt_basic(observation, generate_thought=False, thought=thought)

    def extract_action_from_response(self, response):
        valid_actions = ['move left', 'move right', 'move up', 'move down']
 
        response_lower = response.lower()
        is_random_action = False

        for action in valid_actions:
            if action in response_lower:
                return action, is_random_action

        is_random_action = True
        return random.choice(valid_actions), is_random_action

    def generate_action(self, observation):

        thought_text = None
        if self.generate_thought:
            # analysis step
            thought_prompt = self.compose_thought_prompt(observation)
            thought_input_tensor = thought_prompt.to(self.device)

            with torch.no_grad():
                thought_outputs = self.model.generate(thought_input_tensor, max_new_tokens=128) #20, 50, 64, 128

            thought_tokens = thought_outputs[:, thought_input_tensor.shape[1]:]
            # thought_text: is the LLM thought
            thought_text = self.tokenizer.decode(thought_tokens[0], skip_special_tokens=True).lower()

        # decision step
        prompt = self.compose_prompt(observation, thought_text)
        input_tensor = prompt.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(input_tensor, max_new_tokens=128) #20, 50, 64, 128
        new_tokens = outputs[:, input_tensor.shape[1]:]
        # action_text: is the LLM decision
        action_text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True).lower()


        chosen_action, is_random_action = self.extract_action_from_response(action_text)

        if self.history_window_size != 0:
            pos, surrounding_desc, _, _ = self.history.pop()
            if thought_text is None:
                thought_text = action_text
            self.history.append((pos, surrounding_desc, thought_text, chosen_action))
            if len(self.history) > self.history_window_size:
                self.history.pop(0)

        if self.debug:
            msg = f"\t[LLM Thought]: {thought_text}" + f"\n\n\t[LLM Proposed Action]: {action_text}" + f"\n\n\t[Chosen Action] {chosen_action} (Random Action: {is_random_action})"
            print(msg)

        return chosen_action

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
    def __init__(self, learning_rate, final_learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor, update_type, map_config, is_slippery, full_map_observable, llm_model_name, history_window_size, milestones, full_map_desc_type, generate_thought, debug):
        self.debug = debug
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
        self.llm_model = llmModel(llm_model_name, map_config, is_slippery, full_map_observable, history_window_size, milestones, full_map_desc_type, generate_thought, debug)
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
                text_action = agent.llm_model.generate_action(obs)
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


    args = parser.parse_args()

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

    # naming format of the result folder: env_map/llm/prompt/train_eval/seed 
    agent_name = agent_type_short_names[args.update_type]
    env_id = f"{env_name}_map{map_name}_" + ("IsSlippery" if args.is_slippery else "NotSlippery")
    
    h_tag = f"H{args.history_window_size}"
    o_tag = f"FM{args.full_map_desc_type}" if args.full_map_observable else "PM"
    prompt_tag = f"basic_{h_tag}_{o_tag}"
    if len(milestones) > 0:
        prompt_tag += "_UseMilestones"
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
        full_map_observable = args.full_map_observable,
        llm_model_name = llm_model_name,
        history_window_size = args.history_window_size,
        milestones = milestones,
        full_map_desc_type = args.full_map_desc_type,
        generate_thought = args.generate_thought,
        debug = debug,
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

