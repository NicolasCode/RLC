'''
Classes for implementing the learning methods for continuum state spaces
and discrete action spaces using an approximation function for Q values.
'''
import numpy as np
import random
import torch
from copy import deepcopy
from collections import deque
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class AgentCS :
    '''
    Super class of agents.
    '''

    def __init__(self, parameters:dict, Q):
        self.parameters = parameters
        self.numDims = self.parameters['numDims']
        self.nA = self.parameters['nA']
        self.gamma = self.parameters['gamma']
        self.epsilon = self.parameters['epsilon']
        self.alpha = self.parameters['alpha']
        self.states = deque(maxlen=32)
        self.actions = deque(maxlen=32)
        self.total_reward = 0
        self.rewards = deque(maxlen=32)
        self.rewards.append(np.nan)
        self.dones = deque(maxlen=32)
        self.dones.append(False)
        assert(hasattr(Q, 'predict')), 'Q must be an object with a predict() method'
        assert(hasattr(Q, 'learn')), 'Q must be an object with a learn() method'
        assert(hasattr(Q, 'reset')), 'Q must be an object with a reset() method'
        self.Q = Q
        self.numRound = 0

    def make_decision(self, state=None):
        '''
        Agent makes an epsilon greedy accion based on Q values.
        '''
        epsilon = self._findEpsilon()
        # epsilon = self.epsilon

        if random.uniform(0,1) < epsilon:
            return random.choice(range(self.nA))
        else:
            if state is None:
                state = self.states[-1]
            return self.argmaxQ(state)        

    def _findEpsilon(self):
        if self.epsilon is None:
                
            self.numRound += 1

            parameter = 100/5
            
            if self.numRound < parameter*1:
                return 0.6
            elif self.numRound < parameter*2:
                return 0.3 
            elif self.numRound < parameter*3:
                return 0.15
            elif self. numRound < parameter*4:
                return 0.075
            else:
                return 0
            
        else:
            return self.epsilon
        
    def restart(self):
        '''
        Restarts the agent for a new trial.
        Keeps the same Q for more learning.
        '''
        self.numRound = 0
        self.states = deque(maxlen=32)
        self.actions = deque(maxlen=32)
        self.rewards = deque(maxlen=32)
        self.rewards.append(np.nan)
        self.total_reward = 0
        self.dones = deque(maxlen=32)
        self.dones.append(False)

    def reset(self):
        '''
        Resets the agent for a new simulation.
        '''
        self.restart()
        # Resets the Q function
        self.Q.reset()

    def argmaxQ(self, s):
        '''
        Determines the action with max Q value in state s.
        Breaks ties randomly.
        '''
        # Determines Q values for all actions
        Qs = [self.Q.predict(s, a) for a in range(self.nA)]
        # Determines max Q
        maxQ = max(Qs)
        # Determines ties with maxQ
        opt_acts = [i for i, q in enumerate(Qs) if q == maxQ]
        assert(len(opt_acts) > 0), f'Something wrong with Q function. No maxQ found (Qs={Qs})'
        # Breaks ties uniformly
        return random.choice(opt_acts)

    def update(self, next_state, reward, done):
        '''
        Agent updates its model according to the model.
        TO BE OVERWRITTEN BY SUBCLASS
        '''
        pass

class TargetNetwork(AgentCS):
    '''
    Target 
    '''    
    def __init__(self, parameters:dict, Q):
        super().__init__(parameters, Q)

class SarsaCS(AgentCS) :
    '''
    Implements a SARSA learning rule.
    '''
    def __init__(self, parameters:dict, Q):
        super().__init__(parameters, Q)

    def update(self, next_state, reward, done):
        '''
        Agent updates its model according to the SARSA rule.
        '''
        # Determine current state and action
        state, action = self.states[-1], self.actions[-1]
        if done:
            # Episode is finished. No need to bootstrap update
            G = reward
        else:
            # Episode is active. Bootstraps update
            next_action = self.make_decision(next_state)
            G = reward + self.gamma * self.Q.predict(next_state, next_action)
        # Update weights
        self.Q.learn(state, action, G, self.alpha)




class nStepCS(AgentCS) :
    '''
    Implements a n-step SARSA learning rule.
    '''

    def __init__(self, parameters:dict, Q):
        super().__init__(parameters, Q)
        self.n = self.parameters['n']
        assert(self.n > 0)
        self.T = np.infty
        self.t = 0
        self.tau = 0
   
    def restart(self):
        '''
        Restarts the agent for a new trial.
        Keeps the same Q for more learning.
        '''
        super().restart()
        self.T = np.infty
        self.t = 0
        self.tau = 0

    def update(self, next_state, reward, done):
        '''
        Agent updates its model.
        '''
        def update_tau(tau):
            '''
            - Finds the utility G_{tau:tau+n} 
            - Updates Q[s_tau, a_tau]
            '''
            if tau >= 0:
                # Find utility G_{tau:tau+n} 
                end = min(tau+self.n, self.T)
                G = 0
                # print('\t===>', tau+1, end+1)
                for i in range(tau+1, end+1):
                    discount = self.gamma**(i-tau-1)
                    # print(f'\t\t---{i}---{discount}----')
                    G += discount*rewards[i]  
                # print('\t--->>> G', G)
                if tau + self.n < self.T:
                    s = states[tau + self.n]
                    try:
                        a = self.actions[tau + self.n]
                    except:
                        a = self.make_decision(state=s)
                    # Find bootstrap                    
                    G += self.gamma**self.n * self.Q.predict(s, a)      
                    # print('\tBootstrapping...', G)
                state = states[tau]
                action = self.actions[tau]
                # Update weights
                self.Q.learn(state, action, G, self.alpha)

        # Maintain working list of states and rewards
        states = self.states.append(next_state)
        rewards = self.rewards.append(reward)
        # Check end of episode
        if done:
            # Update T with t + 1 
            self.T = self.t + 1
            # print('Done\n', 'tau', self.tau, 't', self.t, 'T', self.T)
            # Update using remaining information
            for tau in range(self.tau, self.T - 1):
                update_tau(tau)
        else:
            # Set counter to present minus n rounds
            self.tau = self.t - self.n + 1
            # print('tau', self.tau, 't', self.t, 'T', self.T)
            # If present is greater than n, update
            if self.tau >= 0:
                update_tau(self.tau)
            # Update t for next iteration
            self.t += 1



class OnlineQN(SarsaCS) :
    '''
    Implements a SARSA learning rule with a neural network.
    '''
    def __init__(self, parameters:dict, Q):
        super().__init__(parameters, Q)

    def argmaxQ(self, state):
        '''
        Determines the action with max Q value in state s.
        Breaks ties randomly.
        '''
        # Determines Q values for all actions
        with torch.no_grad():
            # Gets predicted Q values
            Qs = self.Q.model(torch.from_numpy(state).float())
            if len(Qs.shape) > 1:
                Qs = Qs[0] 
            # Transforms to list
            Qs = Qs.data.numpy() 
        # Determines max Q
        maxQ = max(Qs)
        # Determines ties with maxQ
        opt_acts = [a for a in range(self.nA) if Qs[a] == maxQ]
        # Breaks ties uniformly
        try:
            return random.choice(opt_acts)
        except:
            print(opt_acts, maxQ)
            raise Exception('Oops')

class ExperienceDataset(Dataset):
    def __init__(self, agent, next_state, reward, done):
        self.agent = agent

        states = [state for state in agent.states]
        self.states = np.array(states)
        self.actions = [action for action in agent.actions]

        next_states = [agent.states[i] for i in range(1, len(agent.states))] + [next_state]
        next_states = np.array(next_states)
        rewards = [agent.rewards[i] for i in range(1, len(agent.rewards))] + [reward]
        dones = [agent.dones[i] for i in range(1, len(agent.dones))] + [done]

        self.updates = [self.get_update(next_states[i], rewards[i], dones[i]) for i in range(len(self.states))]

    def get_update(self, next_state, reward, done):
        if done:
            # Episode is finished. No need to bootstrap update
            G = reward
        else:
            # Episode is active. Bootstrap update
            next_action = self.agent.make_decision(next_state)
            G = reward + self.agent.gamma * self.agent.Q_hat.predict(next_state, next_action)
        return G    

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.updates[idx]
    
class DQN(AgentCS) :
    '''
    Implements the Deep Q Network with 
    experience replay and target network.
    '''
    def __init__(self, parameters:dict, Q):
        super().__init__(parameters, Q)
        self.c = parameters["c"]
        self.len_exp = parameters["len_exp"]
        self.Q_hat = deepcopy(Q)
        self.len_mini_batch = parameters["len_mini_batch"]

    def update(self, next_state, reward, done):
        '''
        Agent updates Q with experience replay and updates target Q.
        '''
        n = len(self.actions)
        k = self.len_exp
        # Obtain indices for batch of experience
        if n < k:
            #agent only learns with the enough experience
            pass
        else:
            # mask = random.sample(range(n), self.len_mini_batch)
            # Get batch of experience
            
            # print(f'reward -> {reward}')
            # print(f'done -> {done}')
            ds_loader = self.create_DataLoader(next_state, reward, done)
            for batch_states, batch_actions, batch_updates in ds_loader:
                # Update weights with batch
                # print(f" batch updates -> {batch_updates}")
                self.Q.learn(batch_states, batch_actions, batch_updates, self.alpha)
            # Check if it's turn to update the target network
            if len(self.actions) % self.c == 0:
                self.Q_hat = deepcopy(self.Q)

    def create_DataLoader(self, next_state, reward, done):
        '''
        Creates a DataLoader object with the experience of the agent.
        '''
        # Create dataset
        ds = ExperienceDataset(self, next_state, reward, done)
        # Create DataLoader
        ds_loader = DataLoader(ds, batch_size=self.len_mini_batch, shuffle=True)
        return ds_loader

    def create_batch(self, mask:list, next_state, reward, done):
        '''
        Creates the training batch.
        Input:
            - mask, a list of indices
            - next_state, the new state obtained the present round
            - reward, the reward obtained the present round
            - done, whether the environment is finished
        Output:
            - batch_states, a list of states
            - batch_actions, the list of corresponding actions
            - batch_updates, the corresponding list of updates
        '''
        # Create the batch of states
        batch_states = [self.states[i][0] for i in range(len(self.states)) if i in mask]
        batch_states = np.array(batch_states)
        # print('batch_states:', batch_states)
        # Get the batch of actions
        batch_actions = [self.actions[i] for i in range(len(self.actions)) if i in mask]
        # print('batch_actions:', batch_actions)
        # Get the updates for each corresponding action
        states_ = [i for i in self.states] + [next_state]
        batch_next_states = [states_[i+1] for i in range(len(self.states)) if i in mask]
        rewards_ = [i for i in self.rewards] + [reward]
        batch_rewards = [rewards_[i+1] for i in range(len(self.states)) if i in mask]
        dones_ = [i for i in self.dones] + [done]
        batch_dones = [dones_[i+1] for i in range(len(self.states)) if i in mask]
        batch_updates = [self.get_update(batch_next_states[i], batch_rewards[i], batch_dones[i]) for i in range(len(mask))]
        # batch_updates = torch.Tensor([batch_updates]).squeeze().detach()
        # print('batch_updates:', batch_updates)
        return batch_states, batch_actions, batch_updates



    def argmaxQ(self, state):
        '''
        Determines the action with max Q value in state s.
        Breaks ties randomly.
        '''
        # Determines Q values for all actions
        with torch.no_grad():
            # Gets predicted Q values
            Qs = self.Q_hat.model(torch.from_numpy(state).float())
            if len(Qs.shape) > 1:
                Qs = Qs[0] 
            # Transforms to list
            Qs = Qs.data.numpy() 
        # Determines max Q
        maxQ = max(Qs)
        # Determines ties with maxQ
        opt_acts = [a for a in range(self.nA) if Qs[a] == maxQ]
        # Breaks ties uniformly
        try:
            return random.choice(opt_acts)
        except:
            print(opt_acts, maxQ)
            raise Exception('Oops')
        
    def reset(self):
        super().reset()
        self.Q_hat.reset()
