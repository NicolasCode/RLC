'''
Module with the state interpreters.
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

def id_state(state):
    '''
    Default interpreter: do nothing.
    '''
    return state

def gym_interpreter1(state):
    '''
    Cleans the state and get only the state space.
    When states come from gymnasium, they contain 
    additional info besides the state space.
    '''
    if isinstance(state, tuple):
        if isinstance(state[1], dict):
            state = state[0]
        else:
            state = state
    else:
        state = state
    return state

def gym_interpreter2(state):
    '''
    Cleans the state and get only the state space.
    When states come from gymnasium, they contain 
    additional info besides the state space. 
    '''
    if isinstance(state,tuple):
        state = state[0]

    #print(state.shape)
    state = cv2.resize(state, [32,32])    
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = state/ 255
    #plt.imshow(state, cmap="gray")
    state = np.expand_dims(state, axis=0)
    return state   

def gym_interpreter3(states):
    '''
    Cleans the state and get only the state space.
    When states come from gymnasium, they contain 
    additional info besides the state space. 
    '''
    last_state = states[1]
    state = states[0]

    if isinstance(state,tuple):
        state = state[0]


    #print(state.shape)
    state = cv2.resize(state, [32,32])    
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = state/ 255
    #plt.imshow(state, cmap="gray")
    state = np.expand_dims(state, axis=0)
    
    if last_state is None:
        return state
    else:

        state = np.add(0.4*last_state, state) 
        # Superponer la imagen en stack sobre state
        return state
    
    '''
        stack = (state, state)
    else:
        stack = (stack[1], state)

    return stack   
    '''

class gym_interpreter_3:

    def __init__(self):
        self.last_state = np.zeros((32,32))
        self.counter = 0
        self.interval = 10

    def interpret(self, state):
        if isinstance(state,tuple):
            state = state[0]
    
        state = cv2.resize(state, [32,32])    
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = state/ 255
        
        if (self.counter % self.interval) == 0:
            
            state = np.add(0.4*self.last_state, state)

            self.last_state = state
            self.counter += 1

        return np.expand_dims(state, axis=0)


def gridW_nS_interpreter(state):
    '''
    Interprets Gridworld state as a ravel index.
    '''
    shape = (state.shape[1], state.shape[2])
    comps = np.where(state == 1)
    to_ravel = [(comps[1][i],comps[2][i]) for i in range(len(comps[0]))]
    ravels = [np.ravel_multi_index(mi, shape) for mi in to_ravel]
    n = np.product(shape)
    n_shape = (n, n, n, n)
    return np.ravel_multi_index(ravels, n_shape)

def gridW_cs_interpreter(state):
    '''
    Interprets Gridworld state as a triple.
    '''
    shape = (state.shape[1], state.shape[2])
    comps = np.where(state == 1)
    to_ravel = [(comps[1][i],comps[2][i]) for i in range(len(comps[0]))]
    ravels = [np.ravel_multi_index(mi, shape) for mi in to_ravel]
    return tuple(ravels)

def gridW_vector_interpreter(state):
    '''
    Interprets Gridworld state as a single vector
    '''
    shape = np.product(state.shape)
    return state.reshape(1, shape)