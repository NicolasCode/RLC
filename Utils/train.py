import Utils.utils as utils
from os import path
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import Utils.track_creating as tc

class TrainRun :
    '''
    Class to train and run an agent in an environment.
    '''
    def __init__(self,\
                env_name:str,\
                state_interpreter,\
                agent,\
                model_name:str,\
                num_rounds:int,\
                num_episodes:int 
                ) -> None:
        self.env_name = env_name
        self.state_interpreter = state_interpreter
        self.agent = agent
        self.num_rounds = num_rounds
        self.num_episodes = num_episodes
        self.model_name = model_name
        file_name = (f'{model_name}_in_{env_name}')
        self.file_model = path.join('models', f'{file_name}.pt')
        self.file_csv = path.join('data', f'{file_name}.csv')
        self.file_png = path.join('images', f'{file_name}.png')
        self.file_losses = path.join('images', f'{file_name}_losses.png')
        self.file_test = path.join('images', f'{file_name}_test.png')
        self.file_compare_hist = path.join('images', f'comparison_hist.png')
        self.file_compare_rew = path.join('images', f'comparison_rew.png')
        self.file_sweep = path.join('images', f'{file_name}_sweep.png')

    def load_env(self, render_mode):
        '''
        Loads environment. If using gymnasium environments, render mode
        is different for training (None) than for running (rgb_array). Render
        mode can only set when instantiating object.
        '''
        if self.env_name in ['MountainCar-v0']:
            self.environment = gym.make(self.env_name,\
                                        render_mode=render_mode
                                          )
        elif self.env_name in ['CarRacing-v2']:
            self.environment = gym.make(self.env_name,\
                                        render_mode=render_mode,\
                                        continuous=False,
                                        lap_complete_percent = 0.08,\
                                        domain_randomize = False,                                       
                                        )
            
            special_step = self.environment.step
            def step(action):
              result = self.environment.special_step(action)

              state = result[0]
              reward = result[1]
              done = result[2]

              # Modificación de reward y done
              # reward = min(0, reward)
              reward = -1
              percent = (self.environment.tile_visited_count)/len(self.environment.track)  
              target = self.environment.lap_complete_percent
              done = done or (percent >= target)
              # points = [target, target/2, target/4]
              # print(self.environment.clock)
              if (percent >= target):
                 reward = 10

              return (state,reward,done,result[3],result[4])
                                          
            special_reset = self.environment.reset
            def reset():
              state = self.environment.special_reset(options={"randomize":True}, seed=1)
              for i in range(90):
                 self.environment.step(3)
              for i in range(10):
                 self.environment.step(4)
              state = self.environment.step(4)
              return state
              
            setattr(self.environment, 'special_step', special_step)
            setattr(self.environment, 'step', step)
            setattr(self.environment, 'special_reset', special_reset)
            setattr(self.environment, 'reset', reset)
            

            # Modificación de _create_track
            def _create_track():
              #res = self.environment.special_create_track()
              print("modificamos")
              return tc._create_track()

            setattr(self.environment, '_create_track', _create_track)
            # self.environment.reset()
            # Starting in frame 80

        else:
            print(self.env_name, self.env_name in ['MountainCar-v0'])
            raise Exception('Unknown environment. Please modify TrainRun.load_env() to include it.')

    def save_agent(self):
        '''
        Saves agent model to a file
        '''

        # Save model for when reset is executed
        # torch.save(self.model.state_dict(), self.model_file)
        self.agent.Q.save()
        pass
    
    def train(self):
        '''
        Trains agent.
        '''
        # Creates environment
        self.load_env(render_mode=None)
        # Creates episode
        episode = utils.Episode(environment=self.environment,\
                agent=self.agent,\
                model_name=self.model_name,\
                num_rounds=self.num_rounds,\
                state_interpreter=self.state_interpreter
                  )
        # Run simulation
        df = episode.simulate(num_episodes=self.num_episodes, file=self.file_csv)
        print(f'Data saved to {self.file_csv}')
        # Plot results
        p = utils.Plot(df)
        if self.num_episodes == 1:
          p.plot_round_reward(file=self.file_png)    
        else:
          p.plot_rewards(file=self.file_png) 
        print(f'Plot saved to {self.file_png}')
        # Save losses if agent uses NN
        
        # if hasattr(self.agent.Q, 'losses'):
        #   losses = self.agent.Q.losses
        #   fig, ax = plt.subplots(figsize=(4,3.5))
        #   ax = sns.lineplot(x=range(len(losses)), y=losses)
        #   ax.set_xlabel("Epoch",fontsize=14)
        #   ax.set_ylabel("Loss",fontsize=14)
        #   plt.savefig(self.file_losses, dpi=300, bbox_inches="tight")
        #   print(f'Plot saved to {self.file_losses}')

    def run(self, visual=True, learn=False, fast=True):
        '''
        Runs the agent on the environment and displays the behavior.
        Input:
          - visual,
            True: displays the environment as in a video using environment render
            False: displays the behavioral data in the terminal step by step
        '''
        if visual:
          # Displays the environment as in a video
          # Creates environment
          self.load_env(render_mode='human')
          # Creates episode
          episode = utils.Episode(environment=self.environment,\
                  agent=self.agent,\
                  model_name=self.model_name,\
                num_rounds=self.num_rounds,\
                state_interpreter=self.state_interpreter
                  )
          if fast:
            episode.sleep_time = 0
          episode.renderize(learn=learn)
        else:
          # Displays data information in the terminal
          self.load_env(render_mode=None)
          # Creates episode
          episode = utils.Episode(environment=self.environment,\
                  agent=self.agent,\
                  model_name=self.model_name,\
                num_rounds=self.num_rounds,\
                state_interpreter=self.state_interpreter
                  )
          episode.run(verbose=4, learn=learn)
        print('Number of rounds:', episode.T - 1)
        print('Total reward:', episode.agent.total_reward)
            
    def test(self, to_df=False):
        '''
        Test the trained agent.
        '''
        # Creates environment
        self.load_env(render_mode=None)
        # Creates episode
        episode = utils.Episode(environment=self.environment,\
                agent=self.agent,\
                model_name=self.model_name,\
                num_rounds=self.num_rounds,\
                state_interpreter=self.state_interpreter
                  )
        # Run simulation
        df = episode.simulate(num_episodes=self.num_episodes, learn=False)
        if to_df:
           # return dataframe
          self.data = df
          self.data.to_csv(self.file_csv)
          print(f'Data saved to {self.file_csv}')
        else:
          # Plot results
          p = utils.Plot(df)
          p.plot_histogram_rewards(self.file_test)
          print(f'Plot saved to {self.file_test}')

    def sweep1(self, parameter:str, values:list, to_df=False):
        '''
        Runs a parameter sweep.
        '''
        # Creates environment
        self.load_env(render_mode=None)
        # Creates experiment
        experiment = utils.Experiment(environment=self.environment,\
                state_interpreter=self.state_interpreter,\
                num_rounds=self.num_rounds,\
                num_episodes=self.num_episodes,\
                num_simulations=1000,\
                  )
        # Run sweep
        df = experiment.run_sweep1(agent=self.agent,\
                                   name=self.model_name,\
                                   parameter=parameter,\
                                   values=values,\
                                   measures=['reward']
                                   )
        if to_df:
            # return dataframe
            self.data = df
        else:
            # Plot results
            p = utils.Plot(df)
            p.plot_rewards(self.file_sweep)
            print(f'Plot saved to {self.file_sweep}')


