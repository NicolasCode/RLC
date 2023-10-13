from Utils.train import TrainRun
from Agents.agentsCS import DQN, OnlineQN
from Agents.deepQ import Uniform_testQ, CNN, CNNL
from Agents.BaseAgent import PPOAgent as PPO
from Utils.interpreters import *
from Utils.utils import Plot
import pandas as pd

def test():
    '''
    Shows a random episode of the Mountain Car
    '''
    # Create agent
    agent = load_ppo()
    # agent = load_DQN(from_file=True, epsilon=0)
    # Create train-and-run object
    interpeter = gym_interpreter_3(size=32)
    act = TrainRun(\
        env_name = 'CarRacing-v2',\
        state_interpreter= interpeter,\
        agent=agent,\
        model_name='PPO',\
        num_rounds=250 ,\
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
    agent = load_ppo()
    # Create train-and-run object
    interpeter = gym_interpreter_3(size=32)
    act = TrainRun(\
        env_name = 'CarRacing-v2',\
        state_interpreter=interpeter,\
        agent=agent,\
        model_name='PPO',\
        num_rounds=50 ,\
        num_episodes=100
        )
    # Show the untrained agent
    print('Training agent...')
    act.train()    
    act.run()

    save = True if input("save model? (y/n)  ->  ").lower() == 'y' else False
    if save:
        act.agent.save()

def load_ppo(from_file = False, epsilon = None):
   '''
    Creates a PPO agent with the given parameters
    '''
   
   #set parameters
   parameters = {"numDims":2,\
                "nA":5,\
                "gamma":1,\
                "epsilon":epsilon,\
                "alpha_policy": 0.001,\
                "alpha_value" : 0.001,\
                "ppo_clip_val":0.2,\
                "target_kl_div":0.01,\
                "max_policy_train_iters":80,\
                "value_train_iters":80,\
                "ppo_epochs":16,\
                "len_exp":16,\
                 }
   
#    model = CNN_CarRacingPPO()

   return PPO(parameters)

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
    Q = CNNL(parameters=parameters)
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
    Q = CNNL(parameters=parameters)
    if from_file:
        Q.load()
    # Create and return agent
    return DQN(parameters, Q)