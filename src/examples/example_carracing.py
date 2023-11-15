'''
© Edgar Andrade 2023
Email: edgar.andrade@urosario.edu.co

-----------------------------------------------
Solving Car Racing
-----------------------------------------------
'''
from utils.performer import Performer
from utils.interpreters import gym_rgb_interpreter
from agents.deepQ import Atari32CNNasQ
from utils.plot_utils import PlotQApprox2D

# Create value approximator
NN_parameters = {'nA':5,
                  'alpha':1e-3,
                  'optimizer_class':'Adam'
                 }
value_approximator = Atari32CNNasQ(parameters=NN_parameters)

# Create agent
agent_parameters = {'nA':3,
                    'gamma':1,
                    'epsilon':0.3,
                    'NN':value_approximator,
                    'target_network_latency':1,
                    'len_exp':32,
                    'batch_size':4,
                    'num_epochs':1,
                    }

# Create performer 
env_parameters = {'lap_complete_percent':0.03,
                  'initial_round':40}
state_interpreter = gym_rgb_interpreter()
perf = Performer(env_name='SpecialCarRacing',
                 env_parameters=env_parameters,
                 state_interpreter=state_interpreter,
                 agent_name='agentsNN.DQN',
                 agent_parameters=agent_parameters)


# Define threads
def check_run():
    print('Running check on environment...')
    perf.run(visual=False, to_video=False, num_rounds=10, sleep_time=0)
    print('Done!')

def train():
    print('Training agent on Mountain Car...')
    perf.train(num_rounds=200, num_episodes=1)
    print('Done!')

def test():
    print('Testing agent on Mountain Car...')
    perf.test()
    print('Done!')
