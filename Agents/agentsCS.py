'''
Classes for implementing the learning methods for continuum state spaces
and discrete action spaces using an approximation function for Q values.
'''
import numpy as np
import random
import torch
from copy import deepcopy

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
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [False]
        assert(hasattr(Q, 'predict')), 'Q must be an object with a predict() method'
        assert(hasattr(Q, 'learn')), 'Q must be an object with a learn() method'
        assert(hasattr(Q, 'reset')), 'Q must be an object with a reset() method'
        self.Q = Q

    def make_decision(self, state=None):
        '''
        Agent makes an epsilon greedy accion based on Q values.
        '''
        if random.uniform(0,1) < self.epsilon:
            return random.choice(range(self.nA))
        else:
            if state is None:
                state = self.states[-1]
            return self.argmaxQ(state)        

    def restart(self):
        '''
        Restarts the agent for a new trial.
        Keeps the same Q for more learning.
        '''
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [False]

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
        states = self.states + [next_state]
        rewards = self.rewards + [reward]
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
            # Transforms to list
            Qs = Qs.data.numpy()[0]
            # print('Qs', Qs)
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



class DQN(AgentCS) :
    '''
    Implements the Deep Q Network with 
    experience replay and target network.
    '''
    def __init__(self, parameters:dict, Q):
        super().__init__(parameters, Q)
        self.c = parameters["c"]
        self.len_sample = parameters["len_sample"]
        self.Q_hat = deepcopy(Q)

    def update(self, next_state, reward, done):
        '''
        Agent updates Q with experience replay and updates target Q.
        '''
        n = len(self.actions)
        k = self.len_sample
        # Obtain indices for batch of experience
        if n > k:
            mask = random.sample(range(n), k)
        else:
            mask = list(range(n))
        # Get batch of experience
        batch_states, batch_actions, batch_updates = self.create_batch(mask, next_state, reward, done)
        # Update weights with batch
        self.Q.learn(batch_states, batch_actions, batch_updates, self.alpha)
        # Check if it's turn to update the target network
        if len(self.actions) % self.c == 0:
            self.Q_hat = deepcopy(self.Q)

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
        states_ = self.states + [next_state]
        batch_next_states = [states_[i+1] for i in range(len(self.states)) if i in mask]
        rewards_ = self.rewards + [reward]
        batch_rewards = [rewards_[i+1] for i in range(len(self.states)) if i in mask]
        dones_ = self.dones + [done]
        batch_dones = [dones_[i+1] for i in range(len(self.states)) if i in mask]
        batch_updates = [self.get_update(batch_next_states[i], batch_rewards[i], batch_dones[i]) for i in range(len(mask))]
        # batch_updates = torch.Tensor([batch_updates]).squeeze().detach()
        # print('batch_updates:', batch_updates)
        return batch_states, batch_actions, batch_updates

    def get_update(self, next_state, reward, done):
        if done:
            # Episode is finished. No need to bootstrap update
            G = reward
        else:
            # Episode is active. Bootstrap update
            next_action = self.make_decision(next_state)
            G = reward + self.gamma * self.Q_hat.predict(next_state, next_action)
        return G

    def argmaxQ(self, state):
        '''
        Determines the action with max Q value in state s.
        Breaks ties randomly.
        '''
        # Determines Q values for all actions
        with torch.no_grad():
            # Gets predicted Q values
            Qs = self.Q.model(torch.from_numpy(state).float())
            # Transforms to list
            Qs = Qs.data.numpy()[0]
            # print('Qs', Qs)
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
