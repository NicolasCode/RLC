import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical

DEVICE = 'cpu'

class ActorCriticNetwork(nn.Module):

    def __init__(self, obs_space_size, action_space_size):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_space_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU())
        self.policy_layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_size))
        self.value_layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def value(self, obs):
        z = self.shared_layers(obs)
        value = self.value_layers(z)
        return value

    def policy(self, obs):
        z = self.shared_layers(obs)
        policy_logits = self.policy_layers(z)
        return policy_logits

    def forward(self, obs):
        z = self.shared_layers(obs)
        policy_logits = self.policy_layers(z)
        value = self.value_layers(z)
        return policy_logits, value  
    
class PPOTrainer():

    def __init__(self,
                 actor_critic,
                 ppo_clip_val=0.2,
                 target_kl_div=0.02,
                 max_policy_train_iters=40,
                 value_train_iters=40,
                 policy_lr=3e-4,
                 value_lr=1e-3):
        self.ac = actor_critic
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters
        policy_params = list(self.ac.shared_layers.parameters()) + \
                        list(self.ac.policy_layers.parameters())
        self.policy_optim = optim.Adam(policy_params, lr=policy_lr)
        value_params = list(self.ac.shared_layers.parameters()) + \
                        list(self.ac.value_layers.parameters())
        self.value_optim = optim.Adam(value_params, lr=value_lr)

    def train_policy(self, obs, acts, old_log_probs, gaes):
        for _ in range(self.max_policy_train_iters):
            self.policy_optim.zero_grad()
            new_logits = self.ac.policy(obs)
            new_logits = Categorical(logits=new_logits)
            new_log_probs = new_logits.log_prob(acts)
            policy_ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = policy_ratio.clamp(1 - self.ppo_clip_val,
                                                1 + self.ppo_clip_val)
            clipped_loss = clipped_ratio * gaes
            full_loss = policy_ratio * gaes
            policy_loss = -torch.min(full_loss, clipped_loss).mean()
            policy_loss.backward()
            self.policy_optim.step()
            kl_div = (old_log_probs - new_log_probs).mean()
            if kl_div >= self.target_kl_div:
                break

    def train_value(self, obs, returns):
        for _ in range(self.value_train_iters):
            self.value_optim.zero_grad()
            values = self.ac.value(obs)
            value_loss = (returns - values) ** 2
            value_loss = value_loss.mean()
            value_loss.backward()
            self.value_optim.step()

def discount_rewards(rewards, gamma=0.99):
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards) - 1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])

def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]
    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])
    return np.array(gaes[::-1])

def rollout(model, env, max_steps=1000):
    train_data = [[], [], [], [], []] # obs, act, reward, values, act_log_probs
    obs = env.reset()[0]
    ep_reward = 0
    for _ in range(max_steps):
        obs_ = torch.tensor(np.asarray(obs),
                        dtype=torch.float32,
                        device=DEVICE)
        logits, val = model(obs_)
        act_distribution = Categorical(logits=logits)
        act = act_distribution.sample()
        act_log_prob = act_distribution.log_prob(act).item()
        result = env.step(act.item())
        next_obs, reward, done = result[0], result[1], result[2]
        for i, item in enumerate((obs, act.item(), reward, val.item(), act_log_prob)):
            train_data[i].append(item)
        obs = next_obs
        ep_reward += reward
        if done:
            break
    train_data = [np.asarray(x) for x in train_data]
    train_data[3] = calculate_gaes(train_data[2], train_data[3])
    return train_data, ep_reward

env = gym.make('CartPole-v1')
model = ActorCriticNetwork(env.observation_space.shape[0],
                           env.action_space.n)
model = model.to(DEVICE)
train_data, reward = rollout(model, env)
n_episodes = 200
print_freq = 20
ppo = PPOTrainer(model)
ep_rewards = []
for episode_idx in range(n_episodes):
    train_data, reward = rollout(model, env)
    ep_rewards.append(reward)
    permute_idxs = np.random.permutation(len(train_data[0]))
    obs = torch.tensor(train_data[0][permute_idxs],
                       dtype=torch.float32,
                       device=DEVICE)
    acts = torch.tensor(train_data[1][permute_idxs],
                       dtype=torch.int32,
                       device=DEVICE)
    gaes = torch.tensor(train_data[3][permute_idxs],
                        dtype=torch.float32,
                        device=DEVICE)
    act_log_probs = torch.tensor(train_data[4][permute_idxs],
                                 dtype=torch.float32,
                                 device=DEVICE)
    returns = discount_rewards(train_data[2])[permute_idxs]
    returns = torch.tensor(returns,
                           dtype=torch.float32,
                           device=DEVICE)
    ppo.train_policy(obs, acts, act_log_probs, gaes)
    ppo.train_value(obs, returns)
    if (episode_idx + 1) % print_freq == 0:
        print(f'Episode {episode_idx + 1} | Avg Reward {np.mean(ep_rewards[-print_freq:])}')
    