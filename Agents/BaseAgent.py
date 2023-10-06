import numpy as np
from typing import Dict
import json
from copy import deepcopy
import torch
from torch.distributions.categorical import Categorical

class PPOAgent:
    '''
    Defines the basic methods for PPO agent.
    '''

    def __init__(self, parameters:Dict[any, any], network):
        self.parameters = parameters
        self.nA = self.parameters['nA']
        self.gamma = self.parameters['gamma']
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [np.nan]
        self.net = network
        self.backup_net = deepcopy(self.net)
        self.log_prob = None
        self.vals = []

        parameters_policy = list(self.net.shared_layers.parameters()) + list(self.net.policy_layers.parameters())
        parameters_values = list(self.net.shared_layers.parameters()) + list(self.net.value_layers.parameters())
    
        self.optimizer_policy = torch.optim.Adam(parameters_policy, lr=self.parameters['alpha_policy'])
        self.optimizer_value = torch.optim.Adam(parameters_values, lr=self.parameters['alpha_value'])

    def make_decision(self):
        '''
        Agent makes a decision according to its policy.
        '''
        state = self.states[-1]
        logits, vals = self.net.forward(state)

        act_distribution = Categorical(logits=logits)
        action = act_distribution.sample()
        self.log_prob = act_distribution.log_prob(action).item()
        self.vals = vals
        
        return action

    def restart(self):
        '''
        Restarts the agent for a new trial.
        '''
        self.states = []
        self.actions = []
        self.vals = [] 
        self.rewards = [np.nan]
        self.dones = [np.nan]
        self.log_prob = None

    def reset(self):
        '''
        Resets the agent for a new simulation.
        '''
        self.restart()
        self.net = self.backup_net

    def update(self, next_state, reward, done):
        


    def save(self, file:str) -> None:
        # Serializing json
        dictionary = {'policy':self.policy.tolist(),
                      'Q':self.Q.tolist()}
        json_object = json.dumps(dictionary, indent=4)
        # Writing to file
        with open(file, "w") as outfile:
            outfile.write(json_object)
        outfile.close()

    def load(self, file:str):
        # Opening JSON file
        with open(file, 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
        self.reset()
        self.policy = np.array(json_object['policy'])
        self.Q = np.array(json_object['Q'])
        openfile.close()