import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from utility import map_configs, platform_seeded, env_seeded
import pickle

# Generic CNN Q-network for square FrozenLake maps
class ConvQNetwork(nn.Module):
    def __init__(self, grid_size, num_actions):
        super(ConvQNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # keeps output shape same
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

# Replay buffer
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

# Convert flat index to 2D one-hot grid
def state_to_grid(state, grid_size):
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    row, col = divmod(state, grid_size)
    grid[row, col] = 1.0
    return grid

# Evaluate agent with greedy policy
def evaluate_agent(env, q_net, grid_size, num_episodes=25, device="cpu"):
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            state_grid = torch.tensor(state_to_grid(state, grid_size)).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                action = q_net(state_grid).argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
    return total_reward / num_episodes


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

# DQN training with CNN and flexible map size
def train_dqn(env_name="FrozenLake-v1", map_name="4x4", max_steps=20000, eval_interval=2000, seed=42, buffer_path=""):
    platform_seeded(seed)

    map_config = map_configs[map_name]
    env = gym.make(env_name, is_slippery=False, map_name=map_name, desc=map_config)

    env_seeded(env, seed)

    num_states = env.observation_space.n
    grid_size = int(math.sqrt(num_states))  # assume square grid
    print(f"[map {map_name}]: {grid_size} x {grid_size}. Seed={seed}")
    assert grid_size * grid_size == num_states, "Environment must have square grid layout"
    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_net = ConvQNetwork(grid_size, num_actions).to(device)
    target_q_net = ConvQNetwork(grid_size, num_actions).to(device)
    target_q_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

    buffer_size = max_steps #50000 #100000
    buffer = ReplayBuffer(capacity=buffer_size)

    batch_size = 32 # 64
    gamma = 0.99
    update_freq = 200 # 100, 200, 500
    epsilon = 1.0
    #epsilon_decay = 0.995
    epsilon_min = 0.05

    if buffer_path != "":
        ave_ret_loaded_buffer = load_buffer(buffer_path, buffer)
        epsilon = max(0.2, 1.0 - ave_ret_loaded_buffer) # ave_ret_loaded_buffer is the normalized return score between 0 and 1
        print(f"A previously saved buffer is loaded: average return is {ave_ret_loaded_buffer}, initial epsilon is {epsilon}")

    target_steps = int(max_steps//2)
    epsilon_init = epsilon
    # decay per step
    epsilon_decay = (epsilon_min / epsilon_init) ** (1.0 / target_steps)


    step_count = 0
    eval_returns = []

    pbar = tqdm(total=max_steps, desc="Training Steps")
    while step_count < max_steps:
        state, _ = env.reset()
        done = False
        while not done and step_count < max_steps:
            state_grid = torch.tensor(state_to_grid(state, grid_size)).unsqueeze(0).unsqueeze(0).to(device)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = q_net(state_grid).argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push((state, action, reward, next_state, done))
            state = next_state
            step_count += 1
            pbar.update(1)

            # Training
            if len(buffer) >= batch_size:
                states, actions, rewards_b, next_states, dones = buffer.sample(batch_size)
                states = torch.tensor([state_to_grid(s, grid_size) for s in states]).unsqueeze(1).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_b = torch.FloatTensor(rewards_b).to(device)
                next_states = torch.tensor([state_to_grid(s, grid_size) for s in next_states]).unsqueeze(1).to(device)
                dones = torch.FloatTensor(dones).to(device)

                q_values = q_net(states).gather(1, actions).squeeze()
                next_q_values = target_q_net(next_states).max(1)[0]
                targets = rewards_b + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, targets.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step_count % update_freq == 0:
                target_q_net.load_state_dict(q_net.state_dict())

            if step_count % eval_interval == 0:
                avg_return = evaluate_agent(env, q_net, grid_size, num_episodes=25, device=device)
                print(f"Step {step_count}: Eval Avg Return = {avg_return:.3f}, epsilon = {epsilon}")
                eval_returns.append((step_count, avg_return))

            # decay per step
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    pbar.close()

    # Final evaluation
    final_eval = evaluate_agent(env, q_net, grid_size, num_episodes=25, device=device)
    print(f"\nFinal Evaluation Average Return: {final_eval:.3f}")
    env.close()

    '''
    # Plot
    if eval_returns:
        steps, evals = zip(*eval_returns)
        plt.plot(steps, evals, marker='o')
        plt.title(f"DQN Evaluation Performance (Map {grid_size}x{grid_size})")
        plt.xlabel("Training Steps")
        plt.ylabel("Average Return")
        plt.grid()
        plot_filename = f"dqn_eval_plot_{grid_size}x{grid_size}.png"
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
        plt.show()
    '''
    return eval_returns


# run experiments
#map_name="2x2"
#max_steps=100000 #20000
#eval_interval=1000
#seed=42
#buffer_path=""
#train_dqn(
#    env_name="FrozenLake-v1", map_name=map_name,
#    max_steps=max_steps, eval_interval=eval_interval, seed=seed,
#    buffer_path=buffer_path)
