from __future__ import annotations
from collections import defaultdict
import os
import random

import torch
import numpy as np
import gymnasium as gym
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm



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

def platform_seeded(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def env_seeded(env, seed):
    env.action_space.seed(seed)
    env.observation_space.seed(seed)



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
    def __init__(self, model_name, map_config, is_slippery, full_map_observable=False, history_window_size=0, milestones=[], full_map_desc_type=0, generate_thought=False, debug=False, use_chatgpt=False, chatgpt_client=None, max_new_tokens=128, use_fewshot=False):
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
        self.use_chatgpt=use_chatgpt
        self.max_new_tokens=max_new_tokens
        self.use_fewshot=use_fewshot
        if self.use_chatgpt:
            self.model = chatgpt_client
        else:
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
            print("Loaded pad_token_id:", self.model.generation_config.pad_token_id)


        # Slippery property description
        self.slippery_property_desc = "The frozen surface is slippery. The intended action is executed with a 1/3 probability, while each of the perpendicular actions is also taken with a 1/3 probability."
        if not self.is_slippery:
            self.slippery_property_desc = "The frozen surface is not slippery such that the intended action is executed with a probability of 1."

    def get_few_shot_text(self):
        few_shot_examples = ""
        examples = []
        # 3x3 map
        '''
        SFF
        FFH
        HFG
        '''
        # do not move towards edges and holes
        example_1=(
            "System: You are playing Frozen Lake, a grid-based game. "
            "The objective is to move from the start tile (0, 0) (marked as S) to the goal tile (2, 2) (marked as G) without falling into holes (H) on the grid. "
            "The frozen surface is not slippery such that the intended action is executed with a probability of 1. "
            "At each step, you can take one of the following actions:\n"
            "- 'left': move meft\n"
            "- 'right': move right\n"
            "- 'up': move up\n"
            "- 'down': move down\n"
            "Be careful to avoid holes and aim to reach the goal as quickly as possible.\n"
            f"{self.map_info_desc_for_prompt}\n"
            "\n\n"
            "[History]: \n"
            "\n\n"
            "User: You are currently at position (1, 0), which is a Start tile.\n"
            "Surrounding tiles:\n"
            "- Up: Start\n"
            "- Down: Hole\n"
            "- Left: Edge of the map\n"
            "- Right: Frozen\n"
            "\n\n"
            "What should you do next? Please respond with one of: 'move left', 'move right', 'move up', 'move down'.\n"
            "Response: move right. Down leads to a hole and failure. Left is the map’s edge, resulting in a wasted step. Both right and up directions are safe, so I randomly select right."
        )        
        examples.append(example_1)
        '''
            "Response: move right. Down leads to a hole and failure. Left is the map’s edge, resulting in a wasted step. Both right and up directions are safe, so I randomly select right."


            "Response: move right. The right direction is a frozen surface that is safe to move onto. "
            "The down direction is a hole that will fail the agent if it is on it. "
            "The left direction is the map's edge where the agent will waste a step if it moves towards it. "
            "The up direction is the start location that is safe to move onto. "
            "Since there are two directions (right and up) that are safe and possibly rewarded to move towards, "
            "here I randomly pick one from them, which is the right direction, and hence decide to move right."
       '''


        # the following example for the situation when the history is enable drives the LLM agent to a worse performance, so disable it for now.
        #if self.history_window_size != 0:
        if False:
            example_2=(
                "System: You are playing Frozen Lake, a grid-based game. "
                "The objective is to move from the start tile (0, 0) (marked as S) to the goal tile (2, 2) (marked as G) without falling into holes (H) on the grid. "
                "The frozen surface is not slippery such that the intended action is executed with a probability of 1. "
                "At each step, you can take one of the following actions:\n"
                "- 'left': move meft\n"
                "- 'right': move right\n"
                "- 'up': move up\n"
                "- 'down': move down\n"
                "Be careful to avoid holes and aim to reach the goal as quickly as possible.\n"
                f"{self.map_info_desc_for_prompt}\n"
                "\n\n"
                "[History]: The following is a step-by-step history of the agent’s past decisions, listed in order from the first action to the most recent one.\n"
                "1. At (0,0), saw: Up=E, Down=F, Left=E, Right=F →  Thought: , Action: move right\n"
                "2. At (0,1), saw: Up=E, Down=F, Left=S, Right=F →  Thought: , Action: move right\n"
                "3. At (0,2), saw: Up=E, Down=H, Left=F, Right=E →  Thought: , Action: move left\n"
                "\n\n"
                "User: You are currently at position (0, 1), which is a Frozen tile.\n"
                "Surrounding tiles:\n"
                "- Up: Edget of the map\n"
                "- Down: Frozen\n"
                "- Left: Start\n"
                "- Right: Frozen\n"
                "\n\n"
                "What should you do next? Please respond with one of: 'move left', 'move right', 'move up', 'move down'.\n"
                "Response: move down. The right direction is a frozen surface that is safe to move onto. The down direction is a frozen surface that is safe to move onto. The up direction is the map's edge where the aget will waste a step if it moves towards it. The left direction is the start location that is safe to move onto. Even thought three directions (right, down and left) are safe to move onto, according to the current history of the agent, the down direction has not been explored and has a potential of leading towards the goal location. So, I decide to move down."
            )
            examples.append(example_1)

        few_shot_examples = "=== [Few-Shot Examples] ==="
        for eid, example_text in enumerate(examples, 1):
            few_shot_examples += f"[Example {eid}]\n\n{example_text}\n\n\n"
        few_shot_examples = "=========================="

        return few_shot_examples

    def get_start_and_goal_position(self):
        self.grid_size = len(self.map_config)
        # Find start and goal positions dynamically
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.map_config[row][col] == "S":
                    self.start_position = (row, col)
                if self.map_config[row][col] == "G":
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
        self.map_info_desc_for_prompt = f"The map is fully observable. The map is a {map_rows}x{map_rows} grid as shown below (each tile is represented by a single capital letter):\n{self.map_text_desc}\n{self.tile_meanings_text}"


    def get_system_content_for_prompt(self):
        # few-shot examples
        fewshot_examples = ""
        if self.use_fewshot:
            fewshot_examples = self.get_few_shot_text()

        # used parameters: self.current_start_position, self.current_goal_position, self.slippery_property_desc and self.map_info_desc_for_prompt
        system_content = (
            f"You are playing Frozen Lake, a grid-based game. The objective is to move from the start tile {self.current_start_position} (marked as S) "
            f"to the goal tile {self.current_goal_position} (marked as G) without falling into holes (H) on the grid. {self.slippery_property_desc}\n"
            "At each step, you can take one of the following actions:\n"
            "- 'left': move left\n"
            "- 'right': move right\n"
            "- 'up': move up\n"
            "- 'down': move down\n\n"
            "Be careful to avoid holes and aim to reach the goal as quickly as possible.\n"
            f"{self.map_info_desc_for_prompt}"
        )

        if fewshot_examples != "":
            return f"{fewshot_examples}\n\n\n{system_content}"
        else:
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

        if self.use_chatgpt:
            return messages
        else:
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

    def generate_text_using_llm(self, input):
        if self.use_chatgpt:
            generated_text = self.model.chat.completions.create(
                messages=input,
                model=self.model_name, #"gpt-4o-mini",
            ).choices[0].message.content.strip()
        else:
            input_tensor = input.to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(input_tensor, max_new_tokens=self.max_new_tokens) #20, 50, 64, 128
            new_tokens = outputs[:, input_tensor.shape[1]:]
            # action_text: is the LLM decision
            generated_text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True).lower()
        return generated_text

    def generate_action(self, observation):

        thought_text = None
        if self.generate_thought:
            # analysis step
            thought_prompt = self.compose_thought_prompt(observation)
            thought_text = self.generate_text_using_llm(thought_prompt)

        # decision step
        prompt = self.compose_prompt(observation, thought_text)
        action_text = self.generate_text_using_llm(prompt)

        # extract the action
        chosen_action, is_random_action = self.extract_action_from_response(action_text)

        # update the history
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
    def __init__(self, learning_rate, final_learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor, update_type, map_config, is_slippery, llm_model_name, history_window_size=0, milestones=[], full_map_observable=False, full_map_desc_type=0, generate_thought=False, debug=False, use_chatgpt=False, chatgpt_client=None, max_new_tokens=128, use_fewshot=False):
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
        self.llm_model = llmModel(llm_model_name, map_config, is_slippery, full_map_observable, history_window_size, milestones, full_map_desc_type, generate_thought, debug, use_chatgpt, chatgpt_client, max_new_tokens, use_fewshot)
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

def evaluate_agent(agent, eval_env, n_episodes_eval=10, debug=False):
    #eval_env = gym.make(env_name, is_slippery=is_slippery, map_name=map_name, desc=map_config)
    eval_returns = []

    for episode in tqdm(range(n_episodes_eval)):
        obs, _ = eval_env.reset()
        if debug:
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



def plot_mean_sem(results_dict, label=None, color=None, save_path=None):
    """
    Plot mean and standard error of the mean (SEM) from a dictionary of results.

    Args:
        results_dict (dict): A dictionary where each key is a seed (e.g., 0, 1, 2, ...) and the value is a list
                             of tuples (step, avg_return), all with the same steps.
        label (str): Label for the line in the plot.
        color (str): Optional color for the line and fill.
        save_path (str): If given, saves the figure to this path.
    
    Returns:
        steps (list of int): Shared step values.
        mean_returns (np.ndarray): Mean return at each step.
        sem_returns (np.ndarray): Standard error of the mean at each step.
    """
    # Extract steps from one of the seeds (assumes same across seeds)
    seeds = list(results_dict.keys())
    steps = [step for step, _ in results_dict[seeds[0]]]

    # Extract returns: shape [num_seeds, num_steps]
    returns = np.array([[ret for _, ret in results_dict[seed]] for seed in seeds])

    # Compute mean and SEM
    mean_returns = returns.mean(axis=0)
    sem_returns = returns.std(axis=0, ddof=1) / np.sqrt(len(seeds))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(steps, mean_returns, label=label or "Mean Return", color=color)
    plt.fill_between(steps, mean_returns - sem_returns, mean_returns + sem_returns, alpha=0.3, color=color)
    plt.xlabel("Step")
    plt.ylabel("Average Return")
    plt.title("Evaluation Performance Across Seeds")
    plt.grid(True)
    if label:
        plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

    return steps, mean_returns, sem_returns


def collect_llm_experiences(save_path, env_name = "FrozenLake-v1", map_name="4x4", n_episodes=1000, seed=0, llm_config=None):
    env = gym.make(env_name, is_slippery=False, map_name=map_name, desc=map_configs[map_name])

    if llm_config is None:
        llm_config = {}

    # Unpack defaults with overrides
    llm_agent = FrozenLakeAgent(
        learning_rate=0.9,
        final_learning_rate=0.0001,
        initial_epsilon=0.0,
        epsilon_decay=0.0,
        final_epsilon=0.0,
        discount_factor=0.9,
        update_type="llm",
        map_config=map_configs[map_name],
        is_slippery=False,
        llm_model_name=llm_config.get("llm_model_name", "TheBloke/deepseek-llm-7b-chat-GPTQ"),
        history_window_size=llm_config.get("history_window_size", 0),
        milestones=llm_config.get("milestones", []),
        full_map_observable=llm_config.get("full_map_observable", False),
        full_map_desc_type=llm_config.get("full_map_desc_type", 0),
        generate_thought=llm_config.get("generate_thought", False),
        debug=llm_config.get("debug", False),
        use_chatgpt=llm_config.get("use_chatgpt", False),
        chatgpt_client=llm_config.get("chatgpt_client", None),
        max_new_tokens=llm_config.get("max_new_tokens", 128),
        use_fewshot=llm_config.get("use_fewshot", False),
    )

    buffer = []

    env.reset(seed=seed)
    env_seeded(env, seed)
    step_count = 0
    episode_returns = []
    current_episode_rewards = []
    episode_id = 1
    total_transitions = 0
    #while step_count < n_episodes:
    for episode_id in tqdm(range(n_episodes), desc=f"Collecting {n_episodes} transitions"):
        state, _ = env.reset()
        done = False
        step_count = 0
        while not done:
            if llm_config["debug"]:
                print(f"[episode {episode_id}, step {step_count}]")
            action = llm_agent.get_action(state, use_llm=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            transition = (state, action, reward, next_state, done)
            buffer.append(transition)
            state = next_state if not done else None

            step_count += 1
            current_episode_rewards.append(reward)

        episode_returns.append(sum(current_episode_rewards))
        current_episode_rewards = []
        total_transitions += step_count

    ave_ret = round(sum(episode_returns)/len(episode_returns), 2)
    save_path = save_path[:-4]+f"_{total_transitions}Trans_{ave_ret}AveR.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(buffer, f)

    print(f"Saved {len(buffer)} transitions to {save_path}: total transitions {total_transitions}, average episode return is {ave_ret}")

