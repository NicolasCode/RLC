'''
Classes for implementing the Q functions
'''
import numpy as np
from Agents.tiles3 import IHT, tiles


class UniformQ():
    '''
    Defines a Q function that returns 0 for all state, action pairs
    '''
    def __init__(self, parameters):
        pass

    def predict(self, state, action):
        return 0
    
    def learn(self, state, action, update, alpha):
        pass

    def reset(self):
        pass


class TilesQ():
    '''
    Defines a tile coding linear approximation.
    Input:
        - numDims (int), number of dimensions of the continuous state space
        - numTilings (int), the number of tilings. Should be a power of 2, e.g., 16.
        - numTiles (list), a list with the number of tiles per dimension
        - scaleFactors (list), a list with the normalization factor per dimension
        - maxSize (int), the max number of tiles
        - weights (list), the list of wheights
    '''
    def __init__(self, parameters, 
                 maxSize=2048, 
                 weights=None):
        self.numDims = parameters["numDims"]
        self.numTilings = parameters["numTilings"]
        self.numTiles = parameters["numTiles"]
        self.scaleFactors = parameters["scaleFactors"]
        self.maxSize = maxSize
        self.iht = IHT(self.maxSize)
        if not weights:
            self.weights = np.zeros(self.maxSize)
        else:
            self.weights = weights
        self.active_tiles = []
            
    def my_tiles(self, state, action):
        '''
        Determines the tiles that get activated by the state
        '''
        # Normalizes the state
        scaled_s = self.normalize(state)
        # Rescale for use with `tiles` using numTiles
        rescaled_s = [scaled_s[i]*self.numTiles[i] for i in range(self.numDims)]
        self.active_tiles = tiles(self.iht, self.numTilings, rescaled_s, [action])
        return self.active_tiles
    
    def predict(self, state, action):
        '''
        Returns the sum of the weights corresponding to the active tiles
        '''
        return sum([self.weights[tile] for tile in self.my_tiles(state, action)])

    def learn(self, state, action, update, alpha):
        '''
        Updates its weights.
        '''
        estimate = self.predict(state, action)
        error = update - estimate
        # Gradient is 1 only for active tiles and 0 otherwise
        # Thus only updates weights of active tiles
        self.weights[self.active_tiles] += alpha * error
        
    def reset(self):
        '''
        Resets weights
        '''
        self.iht = IHT(self.maxSize)
        self.weights = np.zeros(self.maxSize)

    def normalize(self, state):
        '''
        Normalizes state. Should perform the following iteration
        scaled_s = []
        for i, scale in enumerate(self.scaleFactors):
            x = scale(state[i], scale["min"], scale["max"])
            scaled_s.append(x)
        I use list comprehension to optimize speed
        '''
        def re_scale(x, min, max):
            return (x - min) / (max - min)

        return [re_scale(state[i], scale["min"], scale["max"]) for i, scale in enumerate(self.scaleFactors)]