from Utils.train import TrainRun
from Agents.agentsCS import DQN, OnlineQN
from Agents.deepQ import Uniform_testQ, CNN
from Utils.interpreters import gym_interpreter2
from Utils.utils import Plot
import pandas as pd

def test():
    '''
    Shows a random episode of the Mountain Car
    '''
    # Create agent
    agent = load_OnlineQN(from_file=True)
    # Create train-and-run object
    act = TrainRun(\
        env_name = 'CarRacing-v2',\
        state_interpreter=gym_interpreter2,\
        agent=agent,\
        model_name='OnlineQN',\
        num_rounds=1500 ,\
        num_episodes=1
        )
    # Show the untrained agent
    print('Showing the untrained agent...')
    act.run()

def run():
    '''
    Shows a episode of trained Racing car
    '''
    pass


def train():
    '''
    Shows a random episode of the Mountain Car
    '''
    # Create agent
    agent = load_OnlineQN()
    # Create train-and-run object
    act = TrainRun(\
        env_name = 'CarRacing-v2',\
        state_interpreter=gym_interpreter2,\
        agent=agent,\
        model_name='OnlineQN',\
        num_rounds=500 ,\
        num_episodes=1000
        )
    # Show the untrained agent
    print('Training agent...')
    act.train()    

    act.agent.Q.save()
    act.run()

def load_OnlineQN(from_file = False):
    '''
    Creates a OnlineQN agent with the given parameters
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
    if from_file:
        Q.load()
    # Create and retur agent
    return OnlineQN(parameters, Q)   

