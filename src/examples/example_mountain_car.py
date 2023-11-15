'''
© Edgar Andrade 2023
Email: edgar.andrade@urosario.edu.co

-----------------------------------------------
Solving Mountain Car
-----------------------------------------------
'''
from utils.performer import Performer
from utils.interpreters import gym_interpreter1
from agents.linearQ import DimIdentity, TilesQ
from utils.plot_utils import PlotQApprox2D

# Create performer over Mountain Car with Q-learning agent
parameters = {"numDims": 2,
              "numTilings": 8,
              "numTiles": [10, 10],
              "scaleFactors": [{"min":-1.2, "max":0.6},
                               {"min":-0.07, "max":0.07}],
              "alpha":0.1}
value_approximator = TilesQ(parameters=parameters,
                            maxSize=2048,
                            weights=None)
agent_parameters = {'nA':3,
                    'nS':2,
                    'gamma':1,
                    'epsilon':0.1,
                    'alpha':0.1,
                    'Q':value_approximator
                    }
env_parameters = {}
perf = Performer(env_name='MountainCar-v0',
                 env_parameters=env_parameters,
                 state_interpreter=gym_interpreter1,
                 agent_name='agentsCS.QLearningCS',
                 agent_parameters=agent_parameters)

# Define threads
def check_run():
    print('Running check on Mountain Car environment...')
    perf.run(visual=False, to_video=True, num_rounds=10, sleep_time=1e-2)
    print('Done!')

def train():
    print('Training agent on Mountain Car...')
    perf.train(num_rounds=500, num_episodes=100)
    print('Done!')

def test():
    print('Testing agent on Mountain Car...')
    perf.test()
    print('Done!')

def compare():
    print('Comparing Q-learning vs SARSA agents on Frozen-Lake...')
    perf.compare_test(agent_vs_name='SARSA',
                      agent_vs_parameters=agent_parameters)
    print('Done!')
