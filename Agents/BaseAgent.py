import numpy as np
from typing import Dict
import json
from copy import deepcopy
from collections import deque
import torch
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset, DataLoader
from networks import CNN_CarRacingPPO


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
        return self.states[idx], self.actions[idx], self.act_log_probs[idx], self.gaes[idx], self.returns[idx]

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
        self.max_policy_train_iters = self.parameters['max_policy_train_iters']
        self.value_train_iters = self.parameters['value_train_iters']
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [np.nan]
        self.act_log_probs = []
        self.vals = []
        self.net = CNN_CarRacingPPO(self.nA)
        self.backup_net = deepcopy(self.net)
        parameters_policy = list(self.net.shared_layers.parameters()) + list(self.net.policy_layers.parameters())
        parameters_values = list(self.net.shared_layers.parameters()) + list(self.net.value_layers.parameters())
        self.policy_optim = torch.optim.Adam(parameters_policy, lr=self.parameters['alpha_policy'])
        self.value_optim = torch.optim.Adam(parameters_values, lr=self.parameters['alpha_value'])

    def make_decision(self):
        '''
        Agent makes a decision according to its policy.
        '''
        state = self.states[-1]
        logits, val = self.net.forward(state)
        act_distribution = Categorical(logits=logits)
        action = act_distribution.sample().item()
        self.act_log_probs.append(act_distribution.log_prob(action).item())
        self.vals.append(val.item())
        return action

    def restart(self):
        '''
        Restarts the agent for a new trial.
        '''
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [np.nan]
        self.vals = [] 
        self.act_log_probs = []

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
            states = self.states + [next_state]
            actions = self.actions
            rewards = self.rewards + [reward]
            values = self.vals
            act_log_probs = self.act_log_probs
            ds_loader = self.create_DataLoader(states, actions, rewards, values, act_log_probs)
            for batch_states, batch_actions, batch_act_log_probs, batch_gaes, batch_returns in ds_loader:
                # Transform tensors to gpu 
                batch_actions = batch_actions.to("cuda" if torch.cuda.is_available() else "cpu")
                batch_states = batch_states.to("cuda" if torch.cuda.is_available() else "cpu")
                batch_updates = batch_updates.to("cuda"  if torch.cuda.is_available() else "cpu")
                # Update weights with batch
                self.train_policy(batch_states, batch_actions, batch_act_log_probs, batch_gaes)
                self.train_value(batch_states, batch_returns)

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

    def save(self, file:str) -> None:
        pass

    def load(self, file:str):
        pass

    def create_DataLoader(self, states, actions, rewards, values, act_log_probs):
        '''
        Creates a DataLoader object with the experience of the agent.
        '''
        # Create dataset
        ds = ExperienceDataset(self, states, actions, rewards, values, act_log_probs)
        # Create DataLoader
        ds_loader = DataLoader(ds, batch_size=self.len_mini_batch, shuffle=True)
        return ds_loader
