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
    agent = load_DQN(from_file=True, epsilon=0)
    # agent = load_DQN(from_file=True, epsilon=0)
    # Create train-and-run object
    interpeter = gym_interpreter_3(size=32)
    act = TrainRun(\
        env_name = 'CarRacing-v2',\
        state_interpreter= interpeter,\
        agent=agent,\
        model_name='DQN',\
        num_rounds=500 ,\
        num_episodes=1
        )
    # Show the untrained agent
    # act.agent.load(name="prototipe2_1")
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
    agent = load_DQN(from_file=False, epsilon=0.5)

    # Create train-and-run object
    interpeter = gym_interpreter_3(size=32)
    act = TrainRun(\
        env_name = 'CarRacing-v2',\
        state_interpreter=interpeter,\
        agent=agent,\
        model_name='DQN_8_percent',\
        num_rounds=250 ,\
        num_episodes=5
        ) # maximun 10 episodes per training
    # Show the untrained agent

    # act.agent.load(name="prototipe2_1")
    print('Training agent...')
    act.train()    
    act.run()

    save = True if input("save model? (y/n)  ->  ").lower() == 'y' else False
    if save:
        # act.agent.save(name="prototipe2_1")
        act.agent.Q.save()

def load_ppo(from_file = False, epsilon = None):
   '''
    Creates a PPO agent with the given parameters
    '''
   
   #set parameters
   parameters = {"numDims":2,\
                "nA":5,\
                "gamma":0.98,\
                "epsilon":epsilon,\
                "alpha_policy": 0.01,\
                "alpha_value" : 0.003,\
                "ppo_clip_val":0.1,\
                "target_kl_div":0.2,\
                "max_policy_train_iters":40,\
                "value_train_iters":40,\
                "len_exp":8,\
                "epochs":300,\
                 } # alpha policy > alpha value
   
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
                  "c": 32,\
                  "len_exp":32,\
                  "len_mini_batch":16,\
                    }
    # Create function to approximate Q
    # Q = Uniform_testQ(parameters=parameters)
    # Q.action = 3 # Agent has to gas
    Q = CNNL(parameters=parameters)
    if from_file:
        Q.load()
    # Create and return agent
    return DQN(parameters, Q)