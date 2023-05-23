from Utils.train import TrainRun
from Agents.agentsCS import DQN 
from Agents.deepQ import Uniform_testQ, CNN
from Utils.interpreters import gym_interpreter2
from Utils.utils import Plot
import pandas as pd

def test():
    '''
    Shows a random episode of the Mountain Car
    '''
    # Create agent
    agent = load_DQN()
    # Create train-and-run object
    act = TrainRun(\
        env_name = 'CarRacing-v2',\
        state_interpreter=gym_interpreter2,\
        agent=agent,\
        model_name='DQN',\
        num_rounds=150+80 ,\
        num_episodes=1
        )
    # Show the untrained agent
    print('Showing the untrained agent...')
    act.run()


def load_DQN():
    '''
    Creates a DQN agent with the given parameters
    '''
    # Set parameters
    parameters = {"numDims":2,\
                  "nA":5,\
                  "gamma":1,\
                  "epsilon":0.1,\
                  "alpha":0.1,\
                  "c": 2,\
                  "len_sample":1,\
                    }
    # Create function to approximate Q
    # Q = Uniform_testQ(parameters=parameters)
    # Q.action = 3 # Agent has to gas
    Q = CNN(parameters=parameters)
    # Create and retur agent
    return DQN(parameters, Q)   

