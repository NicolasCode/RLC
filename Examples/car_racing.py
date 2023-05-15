from Utils.train import TrainRun
from Agents.agentsCS import SarsaCS
from Agents.linearQ import UniformQ
from Utils.interpreters import gym_interpreter1
from Utils.utils import Plot
import pandas as pd

def test():
    '''
    Shows a random episode of the Mountain Car
    '''
    # Create agent
    agent = load_SarsaCS()
    # Create train-and-run object
    act = TrainRun(\
        env_name = 'CarRacing-v2',\
        state_interpreter=gym_interpreter1,\
        agent=agent,\
        model_name='Sarsa',\
        num_rounds=150,\
        num_episodes=1
        )
    # Show the untrained agent
    print('Showing the untrained agent...')
    act.run()


def load_SarsaCS():
    '''
    Creates a SarsaCS agent with the given parameters
    '''
    # Set parameters
    parameters = {"numDims":2,\
                  "nA":5,\
                  "gamma":1,\
                  "epsilon":0.1,\
                  "alpha":0.1,\
                  "numTilings":8,\
                  "numTiles":[10, 10],\
                  "scaleFactors":[\
                    {"min":-1.2,\
                    "max":0.6},
                    {"min":-0.07,\
                      "max":0.07}]
                    }
    # Create function to approximate Q
    Q = UniformQ(parameters=parameters)
    # Create and retur agent
    return SarsaCS(parameters, Q)   