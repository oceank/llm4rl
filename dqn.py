import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import math
from tqdm import tqdm
import pickle
from utility import map_configs, platform_seeded, env_seeded

# ===== Tile Encoding and One-Hot Observation Conversion =====
tile_channels = {'S': 0, 'F': 1, 'H': 2, 'G': 3, 'A': 4}  # A = Agent
NUM_CHANNELS = 5

def build_observation_tensor(map_desc, obs_int, observability='partial'):
    n = len(map_desc)
    obs_tensor = np.zeros((NUM_CHANNELS, n, n), dtype=np.float32)

    row, col = divmod(obs_int, n)

    for i in range(n):
        for j in range(n):
            tile = map_desc[i][j]
            if observability == 'full' or (abs(i - row) + abs(j - col) == 1 or (i == row and j == col)):
                obs_tensor[tile_channels[tile], i, j] = 1.0

    # Mark the agent's location in the agent channel
    obs_tensor[tile_channels['A'], row, col] = 1.0
    return obs_tensor

# ===== ConvNet for 3D Observations =====
class ConvQNetwork(nn.Module):
    def __init__(self, grid_size, num_actions):
        super(ConvQNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_size = 64 * grid_size * grid_size
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# ===== Replay Buffer =====
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return zip(*transitions)

    def __len__(self):
        return len(self.buffer)

# ===== Evaluation =====
def evaluate_agent(env, q_net, grid_size, map_desc, observability, num_episodes=25, device="cpu"):
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            obs = build_observation_tensor(map_desc, state, observability)
            state_tensor = torch.tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = q_net(state_tensor).argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
    return total_reward / num_episodes

# ===== Load buffer =====
def load_buffer(buffer_path, buffer):
    with open(buffer_path, "rb") as f:
        transitions = pickle.load(f)

    ep_rets = []
    cur_ep_ret = 0.0
    for t in transitions:
        cur_ep_ret += t[-3]
        if t[-1]:
            ep_rets.append(cur_ep_ret)
            cur_ep_ret = 0.0
        buffer.push(t)

    return sum(ep_rets)/len(ep_rets)

# ===== DQN Training =====
def train_dqn(env_name="FrozenLake-v1", map_name="4x4", max_steps=20000, eval_interval=2000, seed=42, buffer_path="", observability='partial'):
    platform_seeded(seed)
    map_desc = map_configs[map_name]
    env = gym.make(env_name, is_slippery=False, map_name=map_name, desc=map_desc)
    env.reset(seed=seed)
    env_seeded(env, seed)

    num_states = env.observation_space.n
    grid_size = int(math.sqrt(num_states))
    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = ConvQNetwork(grid_size, num_actions).to(device)
    target_q_net = ConvQNetwork(grid_size, num_actions).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

    buffer = ReplayBuffer(capacity=max_steps)
    batch_size = 32
    gamma = 0.99
    update_freq = 200
    epsilon = 1.0
    epsilon_min = 0.05

    if buffer_path != "":
        ave_ret_loaded_buffer = load_buffer(buffer_path, buffer)
        epsilon = max(0.2, 1.0 - ave_ret_loaded_buffer)
        print(f"Buffer loaded. Avg return: {ave_ret_loaded_buffer}, Initial epsilon: {epsilon}")

    epsilon_decay = (epsilon_min / epsilon) ** (1.0 / (max_steps // 2))

    step_count = 0
    eval_returns = []

    pbar = tqdm(total=max_steps, desc="Training Steps")
    while step_count < max_steps:
        state, _ = env.reset()
        done = False
        while not done and step_count < max_steps:
            obs = build_observation_tensor(map_desc, state, observability)
            state_tensor = torch.tensor(obs).unsqueeze(0).to(device)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = q_net(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push((state, action, reward, next_state, done))
            state = next_state
            step_count += 1
            pbar.update(1)

            if len(buffer) >= batch_size:
                states, actions, rewards_b, next_states, dones = buffer.sample(batch_size)

                states_tensor = torch.tensor([
                    build_observation_tensor(map_desc, s, observability) for s in states
                ]).to(device)

                next_states_tensor = torch.tensor([
                    build_observation_tensor(map_desc, s, observability) for s in next_states
                ]).to(device)

                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_b = torch.FloatTensor(rewards_b).to(device)
                dones = torch.FloatTensor(dones).to(device)

                q_values = q_net(states_tensor).gather(1, actions).squeeze()
                next_q_values = target_q_net(next_states_tensor).max(1)[0]
                targets = rewards_b + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, targets.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step_count % update_freq == 0:
                target_q_net.load_state_dict(q_net.state_dict())

            if step_count % eval_interval == 0:
                avg_return = evaluate_agent(env, q_net, grid_size, map_desc, observability, num_episodes=25, device=device)
                print(f"Step {step_count}: Eval Avg Return = {avg_return:.3f}, epsilon = {epsilon}")
                eval_returns.append((step_count, avg_return))

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    pbar.close()
    final_eval = evaluate_agent(env, q_net, grid_size, map_desc, observability, num_episodes=25, device=device)
    print(f"\nFinal Evaluation Average Return: {final_eval:.3f}")
    env.close()
    return eval_returns

