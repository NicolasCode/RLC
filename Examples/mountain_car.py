from Utils.train import TrainRun
from Agents.agentsCS import SarsaCS
from Agents.linearQ import TilesQ
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
        env_name = 'MountainCar-v0',\
        state_interpreter=gym_interpreter1,\
        agent=agent,\
        model_name='Sarsa',\
        num_rounds=150,\
        num_episodes=1
        )
    # Show the untrained agent
    print('Showing the untrained agent...')
    act.run()

    

def train_and_run_Sarsa():
    '''
    Trains a SARSA agent on the Mountain Car
    '''
    # Create agent
    agent = load_SarsaCS()
    # Create train-and-run object
    act = TrainRun(\
        env_name = 'MountainCar-v0',\
        state_interpreter=gym_interpreter1,\
        agent=agent,\
        model_name='Sarsa',\
        num_rounds=1000,\
        num_episodes=1000
        )
    # Train the agent
    print('Training the agent...')
    act.train()
    # Show the trained agent
    print('Showing the trained agent...')
    act.num_rounds = 200
    act.num_episodes = 1
    act.run()
    print('Done!')



def sweep_alpha_Sarsa():
    '''
    Runs a sweep over the alpha parameter for a
    Sarsa agent
    '''
    # Create SarsaCS agent
    agent = load_SarsaCS()
    # Create train-and-run object
    act = TrainRun(\
        env_name = 'MountainCar-v0',\
        state_interpreter=gym_interpreter1,\
        agent=agent,\
        model_name='Sarsa',\
        num_rounds=1000,\
        num_episodes=1000
        )
    # Sweep alpha
    print('Sweeping alpha...')
    alphas = [0.1/8, 0.2/8, 0.5/8]
    act.sweep1(parameter='alpha', values=alphas)



def train_and_compare():
    '''
    Compares the performance of Sarsa and n-Step agents
    '''
    # Create SarsaCS agent
    agent = load_SarsaCS()
    # Create train-and-run object
    act = TrainRun(\
        env_name = 'MountainCar-v0',\
        state_interpreter=gym_interpreter1,\
        agent=agent,\
        model_name='Sarsa',\
        num_rounds=1000,\
        num_episodes=1000
        )
    # Train the SarsaCS agent
    print('Training SARSA agent...')
    act.train()
    # Testing the agent
    print('Testing SARSA agent...')
    act.num_episodes = 100
    act.test(to_df=True)
    df_sarsa = act.data
    # Create agent n-step agent
    agent_DQN = load_n_step()
    # Create train-and-run object
    act = TrainRun(\
        env_name = 'MountainCar-v0',\
        state_interpreter=gym_interpreter1,\
        agent=agent,\
        model_name='n-Step',\
        num_rounds=1000,\
        num_episodes=1000
        )
    # Train the agent
    print('Training n-Step agent...')
    act.train()
    # Testing the agent
    print('Testing OnlineQN agent...')
    act.num_episodes = 100
    act.test(to_df=True)
    df_qn = act.data
    # Compare performances
    df = pd.concat([df_sarsa, df_qn], ignore_index=True)
    p = Plot(df)
    p.plot_histogram_rewards(act.file_compare_hist)
    print(f'Plot saved to {act.file_compare_hist}')
    p.plot_rewards(act.file_compare_rew)
    print(f'Plot saved to {act.file_compare_rew}')



def load_SarsaCS():
    '''
    Creates a SarsaCS agent with the given parameters
    '''
    # Set parameters
    parameters = {"numDims":2,\
                  "nA":3,\
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
    Q = TilesQ(parameters=parameters)
    # Create and retur agent
    return SarsaCS(parameters, Q)    

