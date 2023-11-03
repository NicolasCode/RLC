import numpy as np
from typing import Dict
import json
from copy import deepcopy
from collections import deque
import torch
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset, DataLoader
from Agents.networks import CNN_CarRacingPPO
from os import path
from tqdm import tqdm


class ExperienceDataset(Dataset):
    def __init__(self, states, actions, rewards, values, act_log_probs):
        self.states = deepcopy(states)
        self.actions = deepcopy(actions)
        self.act_log_probs = deepcopy(act_log_probs)
        self.gaes = self.calculate_gaes(rewards, values)
        self.returns = self.discount_rewards(rewards)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        t_states = torch.tensor(self.states[idx] , dtype=torch.float32)
        t_actions = torch.tensor(self.actions[idx] , dtype=torch.int32)
        t_act_log_probs = torch.tensor(self.act_log_probs[idx] , dtype=torch.float32)
        t_gaes = torch.tensor(self.gaes[idx] , dtype=torch.float32)
        t_returns = torch.tensor(self.returns[idx] , dtype=torch.float32)
        
        return t_states, t_actions, t_act_log_probs, t_gaes, t_returns

    def discount_rewards(self, rewards, gamma=0.99):
        new_rewards = [float(rewards[-1])]
        for i in reversed(range(len(rewards) - 1)):
            new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
        return np.array(new_rewards[::-1])

    def calculate_gaes(self, rewards, values, gamma=0.99, decay=0.97):
        next_values = np.concatenate([values[1:], [0]])
        deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]
        gaes = [deltas[-1]]
        for i in reversed(range(len(deltas)-1)):
            gaes.append(deltas[i] + decay * gamma * gaes[-1])
        
        # print(f'rewards? -> {rewards}')
        # print(f'values? -> {values}')
        # print(f'next_values? -> {next_values}')
        # print(f'deltas? -> {deltas}')
        # print(f'gaes? -> {gaes}')
        
        return np.array(gaes[::-1])


class PPOAgent:
    '''
    Defines the basic methods for PPO agent.
    '''

    def __init__(self, parameters:Dict[any, any]):
        self.parameters = parameters
        self.nA = self.parameters['nA']
        self.gamma = self.parameters['gamma']
        self.ppo_clip_val = self.parameters['ppo_clip_val']
        self.target_kl_div = self.parameters['target_kl_div']
        self.epochs = self.parameters['epochs']
        # self.max_policy_train_iters = self.parameters['max_policy_train_iters']
        self.max_policy_train_iters = 1
        # self.value_train_iters = self.parameters['value_train_iters']
        self.value_train_iters = 1
        self.len_mini_batch = self.parameters['len_exp']
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [np.nan]
        self.act_log_probs = []
        self.vals = []
        self.net = CNN_CarRacingPPO(self.nA)
        self.backup_net = deepcopy(self.net)
        parameters_policy = list(self.net.shared_layers.parameters()) + list(self.net.policy_layers.parameters())
        # parameters_values = list(self.net.shared_layers.parameters()) + list(self.net.value_layers.parameters())
        parameters_values = list(self.net.value_layers.parameters()) 
        self.policy_optim = torch.optim.Adam(parameters_policy, lr=self.parameters['alpha_policy'])
        self.value_optim = torch.optim.Adam(parameters_values, lr=self.parameters['alpha_value'])

        self.total_reward = 0

    def make_decision(self):
        '''
        Agent makes a decision according to its policy.
        '''
        state = self.states[-1]
        state = torch.tensor(state, dtype=torch.float32)
        logits, val = self.net.forward(state)
        act_distribution = Categorical(logits=logits)
        action = act_distribution.sample()
        self.act_log_probs.append(act_distribution.log_prob(action).item())
        self.vals.append(val.item())
        return action.item()

    def restart(self):
        '''
        Restarts the agent for a new trial.
        '''
        self.states = []
        self.actions = []
        self.rewards = [-1]
        self.dones = [np.nan]
        self.vals = [] 
        self.act_log_probs = []

        self.total_reward = 0

    def reset(self):
        '''
        Resets the agent for a new simulation.
        '''
        self.restart()
        self.net = deepcopy(self.backup_net)

    def update(self, next_state, reward, done):
        '''
        Agent updates with Monte Carlo simulation and PPO.
        '''
        if done:
            states = self.states #+ [next_state]
            actions = self.actions
            rewards = self.rewards + [reward]
            values = self.vals
            act_log_probs = self.act_log_probs
            # print("lens:")
            # print(len(states), len(actions), len(rewards), len(values), len(act_log_probs))
            ds_loader = self.create_DataLoader(states, actions, rewards[1:], values, act_log_probs)
            for i in tqdm(range(self.epochs)):
                for batch_states, batch_actions, batch_act_log_probs, batch_gaes, batch_returns in ds_loader:
                    # Transform tensors to gpu 
                    batch_actions = batch_actions.to("cuda" if torch.cuda.is_available() else "cpu")
                    batch_states = batch_states.to("cuda" if torch.cuda.is_available() else "cpu")
                    batch_act_log_probs = batch_act_log_probs.to("cuda"  if torch.cuda.is_available() else "cpu")
                    batch_gaes = batch_gaes.to("cuda"  if torch.cuda.is_available() else "cpu")
                    batch_returns = batch_returns.to("cuda"  if torch.cuda.is_available() else "cpu")
                    # Update weights with batch
                    self.train_policy(batch_states, batch_actions, batch_act_log_probs, batch_gaes)
                    self.train_value(batch_states, batch_returns)

    def train_policy(self, obs, acts, old_log_probs, gaes):
        for _ in range(self.max_policy_train_iters):
            self.policy_optim.zero_grad()
            new_logits = self.net.policy(obs)
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
            values = self.net.value(obs)
            value_loss = (returns - values) ** 2
            value_loss = value_loss.mean()
            value_loss.backward()
            self.value_optim.step()

    def save(self, name = "base") -> None:
        torch.save(self.net.state_dict(), path.join('data', f'ppo_{name}_trained.pt'))
    
    def load(self, name = "base"):
        self.net.load_state_dict(torch.load(path.join('data', f'ppo_{name}_trained.pt')))
        self.net.eval()
        
    def create_DataLoader(self, states, actions, rewards, values, act_log_probs):
        '''
        Creates a DataLoader object with the experience of the agent.
        '''
        # Create dataset
        ds = ExperienceDataset(states, actions, rewards, values, act_log_probs)
        # print("len datal" ,len(ds))
        # Create DataLoader
        ds_loader = DataLoader(ds, batch_size=self.len_mini_batch, shuffle=True)
        return ds_loader