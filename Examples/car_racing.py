from Utils.train import TrainRun
from Agents.agentsCS import DQN, OnlineQN
from Agents.deepQ import Uniform_testQ, CNN
from Utils.interpreters import *
from Utils.utils import Plot
import pandas as pd

def test():
    '''
    Shows a random episode of the Mountain Car
    '''
    # Create agent
    agent = load_DQN(from_file=True, epsilon=0)
    # Create train-and-run object
    interpeter = gym_interpreter_3(size=16)
    act = TrainRun(\
        env_name = 'CarRacing-v2',\
        state_interpreter= interpeter,\
        agent=agent,\
        model_name='OnlineQN',\
        num_rounds=500 ,\
        num_episodes=1
        )
    # Show the untrained agent
    print('Showing the untrained agent...')
    act.run(visual=True)
    # act.test(to_df=True)

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
    agent = load_DQN(from_file=True, epsilon=None)
    # Create train-and-run object
    interpeter = gym_interpreter_3(size=16)
    act = TrainRun(\
        env_name = 'CarRacing-v2',\
        state_interpreter=interpeter,\
        agent=agent,\
        model_name='DQN',\
        num_rounds=200 ,\
        num_episodes=500
        )
    # Show the untrained agent
    print('Training agent...')
    act.train()    

    act.agent.Q.save()
    act.run()

def load_OnlineQN(from_file = False, epsilon = None):
    '''
    Creates a OnlineQN agent with the given parameters
    '''
    # Set parameters
    parameters = {"numDims":2,\
                  "nA":5,\
                  "gamma":1,\
                  "epsilon":epsilon,\
                  "alpha":0.1,\
                  "c": 2,\
                  "len_exp":1,\
                    }
    # Create function to approximate Q
    # Q = Uniform_testQ(parameters=parameters)
    # Q.action = 3 # Agent has to gas
    Q = CNN(parameters=parameters)
    if from_file:
        Q.load()
    # Create and retur agent
    return OnlineQN(parameters, Q)   

def load_DQN(from_file = False, epsilon = None):
    '''
    Creates a DQN agent with the given parameters
    '''
    # Set parameters
    parameters = {"numDims":2,\
                  "nA":5,\
                  "gamma":1,\
                  "epsilon":epsilon,\
                  "alpha":0.001,\
                  "c": 16,\
                  "len_exp":16,\
                  "len_mini_batch":8,\
                    }
    # Create function to approximate Q
    # Q = Uniform_testQ(parameters=parameters)
    # Q.action = 3 # Agent has to gas
    Q = CNN(parameters=parameters)
    if from_file:
        Q.load()
    # Create and return agent
    return DQN(parameters, Q)