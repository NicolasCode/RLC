'''
Module with the state interpreters.
'''
import numpy as np

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