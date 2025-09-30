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
                                        continuous=False,\
                                        lap_complete_percent = 0.07,\
                                        domain_randomize = False,                                       
                                        )
            
            special_step = self.environment.step
            special_reset = self.environment.reset
            
            def step(action):
                obs, reward, terminated, truncated, info = self.environment.special_step(action)
                obs = self.state_interpreter.interpret(obs)
                return obs, reward, terminated, truncated, info
            
            def reset(seed=None, options=None):
                obs, info = self.environment.special_reset(seed=seed, options=options)
                obs = self.state_interpreter.interpret(obs)
                return obs, info

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
        Saves agent to file
        '''
        torch.save(self.agent.state_dict(), self.file_model)
        print(f'Agent saved to {self.file_model}')

    def train(self):
        '''
        Trains agent for num_rounds rounds, each round has num_episodes episodes.
        '''
        self.load_env(render_mode=None)
        
        # Initialize lists to store metrics
        episode_rewards = []
        episode_losses = []
        
        for round_num in range(self.num_rounds):
            print(f'Round {round_num + 1}/{self.num_rounds}')
            
            for episode in range(self.num_episodes):
                obs, info = self.environment.reset()
                total_reward = 0
                done = False
                
                while not done:
                    action = self.agent.make_decision(obs)
                    next_obs, reward, terminated, truncated, info = self.environment.step(action)
                    done = terminated or truncated
                    
                    # Store experience and update
                    self.agent.store_experience(obs, action, reward, next_obs, done)
                    self.agent.update(next_obs, reward, done)
                    
                    obs = next_obs
                    total_reward += reward
                
                episode_rewards.append(total_reward)
                print(f'Episode {episode + 1}: Reward = {total_reward:.2f}')
        
        self.environment.close()
        print('Training completed!')

    def run(self, visual=True, learn=False, fast=True):
        '''
        Runs agent for one episode. If visual=True, renders environment.
        If learn=True, agent learns during run.
        '''
        render_mode = 'human' if visual else None
        self.load_env(render_mode=render_mode)
        
        obs, info = self.environment.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        while not done:
            action = self.agent.make_decision(obs)
            next_obs, reward, terminated, truncated, info = self.environment.step(action)
            done = terminated or truncated
            
            if learn:
                self.agent.store_experience(obs, action, reward, next_obs, done)
                self.agent.update(next_obs, reward, done)
            
            obs = next_obs
            total_reward += reward
            step_count += 1
            
            if visual:
                self.environment.render()
        
        print(f'Episode completed: Steps = {step_count}, Total Reward = {total_reward:.2f}')
        self.environment.close()
        return total_reward

    def test(self, to_df=False):
        '''
        Tests agent performance over multiple episodes
        '''
        self.load_env(render_mode=None)
        
        test_rewards = []
        for episode in range(10):  # Test for 10 episodes
            obs, info = self.environment.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.agent.make_decision(obs)
                obs, reward, terminated, truncated, info = self.environment.step(action)
                done = terminated or truncated
                total_reward += reward
            
            test_rewards.append(total_reward)
            print(f'Test Episode {episode + 1}: Reward = {total_reward:.2f}')
        
        avg_reward = np.mean(test_rewards)
        print(f'Average Test Reward: {avg_reward:.2f}')
        
        self.environment.close()
        return test_rewards if to_df else avg_reward

    def sweep1(self, parameter:str, values:list, to_df=False):
        '''
        Performs parameter sweep over given values
        '''
        results = []
        
        for value in values:
            print(f'Testing {parameter} = {value}')
            # Set parameter value
            setattr(self.agent, parameter, value)
            
            # Test performance
            avg_reward = self.test(to_df=False)
            results.append((value, avg_reward))
            
        print('Parameter sweep completed!')
        return results
