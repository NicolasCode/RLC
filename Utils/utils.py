'''
Helper functions to gather, process and visualize data
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import lineplot, histplot
from tqdm import tqdm
from copy import deepcopy
from time import sleep
from IPython.display import clear_output
from tqdm import tqdm
import Utils.interpreters
from Utils.interpreters import id_state
from os import path

class Episode :
    '''
    Runs the environment for a number of rounds and keeps tally of everything.
    '''

    def __init__(self, environment, agent, model_name:str, num_rounds:int, id:int=0, state_interpreter=id_state):
        self.environment = environment
        self.state_interpreter = state_interpreter
        print(self.state_interpreter)
        self.agent = agent
        self.model_name = model_name
        self.num_rounds = num_rounds
        self.done = False
        self.T = 1
        self.id = id
        self.sleep_time = 0.3
        state_ = self.environment.reset()
        state = self.interpret_state(state_)
        self.agent.restart()
        self.agent.states.append(state)

    def play_round(self, verbose:int=0, learn=True):
        '''
        Plays one round of the game.
        Input:
            - verbose, to print information.
                0: no information
                1: only number of simulation
                2: simulation information
                3: simulation and episode information
                4: simulation, episode and round information
        '''
        # Ask agent to make a decision
        action = self.agent.make_decision()
        # Saves the action selected
        self.agent.actions.append(action)
        # Runs the environment and obtains the next_state, reward, done, info
        result = self.environment.step(action)
        next_state = self.interpret_state(result[0])
        reward = result[1]
        done = result[2]
        # Prints info
        if verbose > 3:
            state = self.agent.states[-1]
            # print(f'\tThe state is => {state}')
            print(f'\tAgent takes action => {action}')
            # print(f'\tThe state obtained is => {next_state}')
            print(f'\tThe reward obtained is => {reward}')
            print(f'\tEnvironment is finished? => {done}')
            print(f'\tLap percent => {self.environment.lap_complete_percent * 100} %')
            print(f'\tEnviroment tiles visited => {self.environment.tile_visited_count} ( {self.environment.tile_visited_count/len(self.environment.track) * 100} %)')
            print(f'\tEnviroment total tiles => {len(self.environment.track)}')
            print(f'\tEnviroment new lap => {self.environment.new_lap}')
            # Agent learns
        if learn:
            self.agent.update(next_state, reward, done)
        # Saves results
        self.agent.states.append(next_state)
        self.agent.rewards.append(reward)
        self.agent.dones.append(done)
        # Updates round counter
        self.T += 1
        # Updates environment's running
        self.done = done

    def run(self, verbose:int=0, learn=True):
        '''
        Plays the specified number of rounds.
        '''
        for round in range(self.num_rounds):
            if not self.done:
                if verbose > 2:
                    print('\n' + '-'*10 + f'Round {round}' + '-'*10 + '\n')
                self.play_round(verbose=verbose, learn=learn)                
            else:
                break
        return self.to_pandas()
        #return pd.DataFrame()

    def to_pandas(self) -> pd.DataFrame:
        '''
        Creates a pandas dataframe with the information from the current objects.
        Output:
            - pandas dataframe with the following variables:           
                Variables:
                    * episode: a unique identifier for the episode
                    * round: the round number
                    * action: the player's action
                    * reward: the player's reward
                    * done: whether the environment is active or not
                    * model: the model's name
        '''
        # Include las item in actions list
        self.agent.actions.append(np.nan)
        # n1 = len(self.agent.states)
        # n2 = len(self.agent.actions)
        # n3 = len(self.agent.rewards)
        # n4 = len(self.agent.dones)
        # print(n1, n2, n3, self.T)
        data = {}
        data["episode"] = []
        data["round"] = []
        data["state"] = []
        data["action"] = []
        data["reward"] = []
        data["done"] = []
        # for r in range(self.T):
        for r in range(len(self.agent.rewards)):
            data["episode"].append(self.id)
            data["round"].append(r)
            data["state"].append(self.agent.states[r])
            data["action"].append(self.agent.actions[r])
            data["reward"].append(self.agent.rewards[r])
            data["done"].append(self.agent.dones[r])
        df = pd.DataFrame.from_dict(data)        
        df["model"] = self.model_name
        return df

    def reset(self):
        '''
        Reset the episode. This entails:
            reset the environment
            restart the agent 
                  (new states, actions and rewards, 
                   but keep Q and policy)
        '''
        state = self.environment.reset()
        state = self.interpret_state(state)
        self.agent.restart()
        self.agent.states.append(state)
        self.T = 1
        self.done = False

    def renderize(self, parameters=None, learn=True):
        '''
        Plays the specified number of rounds.
        '''
        for round in range(self.num_rounds):
            if not self.done:
                self.play_round(verbose=0, learn=learn)                
                clear_output(wait=True)
                self.environment.render()
                sleep(self.sleep_time)
            else:
                break
   
    def simulate(self, num_episodes:int=1, file:str=None, verbose:int=0, learn=True):
        '''
        Runs the specified number of episodes for the given number of rounds.
        Input:
            - num_episodes, int with the number of episodes.
            - file, string with the name of file to save the data on.
            - verbose, to print information.
                0: no information
                1: only number of simulation
                2: simulation information
                3: simulation and episode information
                4: simulation, episode and round information
        Output:
            - Pandas dataframe with the following variables:

                Variables:
                    * id_sim: a unique identifier for the simulation
                    * round: the round number
                    * action: the player's action
                    * reward: the player's reward
                    * done: whether the environment is active or not
                    * model: the model's name
        '''
        # Create the list of dataframes
        data_frames = []
        # Run the number of episodes
        for ep in tqdm(range(num_episodes)):
            if verbose > 1:
                print('\n' + '='*10 + f'Episode {ep}' + '='*10 + '\n')
            # Reset the episode
            self.reset()
            self.id = ep
            # Run the episode
            df = self.run(verbose=verbose, learn=learn)
            # print(self.agent.Q)
            # Include episode in list of dataframes
            data_frames.append(df)
        # Concatenate dataframes
        data = pd.concat(data_frames, ignore_index=True)
        if file is not None:
            data.to_csv(file)
        return data

    def interpret_state(self, state):
        '''
        Interprets the state to do one or several tasks
        of reshaping and cleaning additional info
        '''
        return self.state_interpreter.interpret(state)

class Plot :
    '''
    Gathers a number of frequently used visualizations.
    '''

    def __init__(self, data:pd.DataFrame):
        self.data = data

    def plot_rewards(self, file:str=None) -> plt.axis:
        '''
        Plots the reward per episode.
        Input:
            - file, string with the name of file to save the plot on.
        Output:
            - axis, a plt object, or None.
        '''
        models = self.data.model.unique()
        vs_models = True if len(models) > 1 else False
        fig, ax = plt.subplots(figsize=(4,3.5))
        data = self.data.copy()
        if 'simulation' in data.columns:
            data = data.groupby(['model', 'simulation', 'episode'])['reward'].sum().reset_index()
        else:
            data = data.groupby(['model', 'episode'])['reward'].sum().reset_index()
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total reward')
        ax.grid()
        if vs_models:
            try:
                ax = lineplot(x='episode', y='reward', hue='model', data=data, errorbar=('ci', 95))
            except:
                ax = lineplot(x='episode', y='reward', hue='model', data=data)
        else:
            try:
                ax = lineplot(x='episode', y='reward', data=data, errorbar=('ci', 95))
            except:
                ax = lineplot(x='episode', y='reward', data=data)
        if file is not None:
            plt.savefig(file, dpi=300, bbox_inches="tight")
        return ax
    
    def plot_round_reward(self, file:str=None) -> plt.axis:
        '''
        Plots the reward per round, averaged over episodes.
        Input:
            - file, string with the name of file to save the plot on.
        Output:
            - axis, a plt object, or None.
        '''
        models = self.data.model.unique()
        vs_models = True if len(models) > 1 else False
        fig, ax = plt.subplots(figsize=(4,3.5))
        ax.set_xlabel('Round')
        ax.set_ylabel('Av. Reward')
        ax.grid()
        if vs_models:
            try:
                ax = lineplot(x='round', y='reward', hue='model', data=self.data, errorbar=('ci', 95))
            except:
                ax = lineplot(x='round', y='reward', hue='model', data=self.data)
        else:
            try:
                ax = lineplot(x='round', y='reward', data=self.data, errorbar=('ci', 95))
            except:
                ax = lineplot(x='round', y='reward', data=self.data)
        if file is not None:
            plt.savefig(file, dpi=300, bbox_inches="tight")

    def plot_histogram_rewards(self, file:str=None) -> plt.axis:
        '''
        Plots a histogram with the sum of rewards per episode.
        Input:
            - file, string with the name of file to save the plot on.
        Output:
            - axis, a plt object, or None.
        '''
        models = self.data.model.unique()
        vs_models = True if len(models) > 1 else False
        fig, ax = plt.subplots(figsize=(4,3.5))
        ax.set_xlabel('Sum of rewards')
        ax.set_ylabel('Frequency')
        ax.grid()
        if vs_models:
            df = self.data.groupby(['model', 'episode']).reward.sum().reset_index()
            ax = histplot(x='reward', hue='model', data=df)
        else:
            df = self.data.groupby('episode').reward.sum().reset_index()
            ax = histplot(x='reward', data=df)
        if file is not None:
            plt.savefig(file, dpi=300, bbox_inches="tight")
        df = self.data.groupby(['model','episode']).reward.sum().reset_index()
        total_reward = df.groupby('model').reward.mean()
        print('Average sum of rewards:\n', total_reward)
        df = self.data.groupby(['model','episode']).done.sum().reset_index()
        success = df.groupby('model').done.mean()*100
        print('\nEnvironment finished percentage:\n', success)

    def plot_rewards_sweep2(self, parameter1:str, parameter2:str, file:str=None) -> plt.axis:
        '''
        Plots the average scores according to sweep of two parameters.
        Input:
            - parameter1, string with the first parameter name.
            - parameter2, string with the first parameter name.
            - file, string with the name of file to save the plot on.
        Output:
            - axis, a plt object, or None.
        '''
        # Keep only last 20% of rounds
        num_rounds = max(self.data['round'].unique())
        cut = int(num_rounds * 0.8)
        df = self.data[self.data['round'] > cut].reset_index()
        # Find average score per pair of parameters' values
        df = df.groupby([parameter2, parameter1, 'simulation', 'episode'])['reward'].sum().reset_index()
        df = df.groupby([parameter2, parameter1])['reward'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(4,3.5))
        ax = lineplot(x=parameter1, y='reward', hue=parameter2, marker='o', data=df)
        ax.set_xlabel(parameter1)
        ax.set_ylabel('Av. Reward')
        ax.grid()
        if file is not None:
            plt.savefig(file, dpi=300, bbox_inches="tight")



class Experiment :
    '''
    Compares given models on a number of measures.
    '''

    def __init__(self, \
                environment, \
                state_interpreter,\
                num_rounds:int, \
                num_episodes:int, \
                num_simulations:int):
        '''
        Input:
            - environment, object with the environment on which to test the agents.
            - num_rounds, int with the number of rounds.
            - num_episodes, int with the number of episodes.
            - num_simulations, int with the number of times the environment should be
                restarted and run the episodes again.
        '''
        self.environment = environment
        self.state_interpreter = state_interpreter
        self.num_rounds = num_rounds
        self.num_episodes = num_episodes
        self.num_simulations = num_simulations
        self.data = None

    def run_experiment(self, \
                       agents:list, \
                       names:list, \
                       measures:list, \
                       verbose:int=0):
        '''
        For each agent, runs the stipulated number of simulations,
        obtains the data and shows the plots on the given measures.
        Input:
            - agents, list of agent objects.
            - names, list of names of the models implemented by the agents.
            - measures, list of measures, which could contain the following strings:
                * 'reward'
                * 'round_reward'
                * 'histogram'
             - verbose, to print information.
                0: no information
                1: only number of simulation
                2: simulation information
                3: simulation and episode information
                4: simulation, episode and round information
        '''
        # Creates the list of dataframes
        data_frames = []
        # Run simulations
        for k in tqdm(range(self.num_simulations)):
            if verbose > 0:
                print('\n' + '*'*10 + f' Simulation {k} ' + '*'*10 + '\n')
            # Reset the agents for new learning history
            for agent in agents:
                agent.reset()
            # Run simulation for each agent
            for i, agent in enumerate(agents):
                if verbose > 0:
                    print('\n' + '%'*10 + f' Agent {names[i]} ' + '%'*10 + '\n')
                # Initialize simulation
                sim = Episode(environment=self.environment, \
                              agent=agent, \
                              model_name=names[i],\
                              num_rounds=self.num_rounds)
                # Run simulation
                df = sim.simulate(num_episodes=self.num_episodes, \
                                  verbose=verbose)
                df['simulation'] = k
                data_frames.append(df)
        # Consolidate data
        data = pd.concat(data_frames, ignore_index=True)
        self.data = data
        # Create plots
        for m in measures:
            file = path.join('images', m)
            if m == 'reward':
                ax = Plot(data).plot_rewards(file)
            if m == 'round_reward':
                ax = Plot(data).plot_round_reward(file)
            if m == 'histogram':
                ax = Plot(data).plot_histogram_rewards(file)

    def run_sweep1(self, \
                       agent, \
                       name:str, \
                       parameter:str, \
                       values:list, \
                       measures:list):
        '''
        For each agent, runs a parameter sweep the simulation the stipulated number
        of times, obtains the data and shows the plots on the given measures.
        Input:
            - agent, an object agent.
            - name, the name of the model implemented by the agent.
            - parameter, a string with the name of the parameter.
            - values, a list with the parameter's values.
            - measures, list of measures, which could contain the following strings:
                * 'reward'
                *  'round_reward'
        '''
        # Creates list of agents
        agents = []
        for value in values:
            agent_ = deepcopy(agent)
            instruction = f'agent_.{parameter} = {value}'
            exec(instruction)
            agents.append(agent_)
        # Creates list of names
        names = [f'({name}) {parameter}={value}' for value in values]
        # Run the simulations
        self.run_experiment(agents=agents,\
                                   names=names,\
                                   measures=measures)

    def run_sweep2(self, \
                       agent, \
                       name:str, \
                       parameter1:str, \
                       values1:list,\
                       parameter2:str, \
                       values2:list,\
                       measures:list, \
                        ):
        '''
        Runs a parameter sweep of one parameter, 
        obtains the data and shows the plots on the given measures.
        Input:
            - agent, an object agent.
            - name, the name of the model implemented by the agent.
            - parameter1, a string with the name of the first parameter.
            - values1, a list with the first parameter's values.
            - parameter2, a string with the name of the second parameter.
            - values2, a list with the second parameter's values.
            - measures, list of measures, which could contain the following strings:
                * 'reward'
        '''
        # Create an auxiliary dataframe to split computational power
        # between RAM and ROM.
        columns = ['model', parameter2, parameter1, 'reward']
        df_aux = pd.DataFrame({column:[] for column in columns})
        temp_file = path.join('data', 'temp_df.csv')
        df_aux.to_csv(temp_file, index=False)
        for value1 in values1:
            # Creates list of agents
            agents = []
            # Creates list of names
            names = []
            for value2 in values2:
                # Creates agent with given parameters
                agent_ = deepcopy(agent)
                instruction = f'agent_.{parameter1} = {value1}'
                exec(instruction)
                instruction = f'agent_.{parameter2} = {value2}'
                exec(instruction)
                agents.append(agent)
                # Creates name
                name_ = f'{name} -- {parameter1}={value1} -- {parameter2}={value2}'
                names.append(name_)
            # Run the simulations
            self.run_experiment(agents=agents,\
                                    names=names,\
                                    measures=[])
            # Provide columns with respective parameter values
            values_p2 = []
            for key, grp in self.data.groupby('model'):
                params = key.split(' -- ')
                value = params[2].split('=')[1]
                values_p2 += [value]*grp.shape[0]
            self.data[parameter1] = value1
            self.data[parameter2] = values_p2
            # Keep only last 20% of rounds
            num_rounds = max(self.data['round'].unique())
            cut = int(num_rounds * 0.8)
            self.data = self.data[self.data['round'] > cut].reset_index()
            # Find average score per pair of parameters' values
            self.data['model'] = name
            self.data = self.data.groupby(['model', parameter2, parameter1, 'simulation', 'episode'])['reward'].sum().reset_index()
            self.data = self.data.groupby(['model', parameter2, parameter1])['reward'].mean().reset_index()
            # Save to disk
            self.data.to_csv(temp_file, mode='a', index=False, header=False)
        # Get the whole dataframe to create the plot
        df_aux = pd.read_csv(temp_file)
        # Plot
        fig, ax = plt.subplots(figsize=(4,3.5))
        ax = lineplot(x=parameter1, y='reward', hue=parameter2, marker='o', data=df_aux)
        ax.set_xlabel(parameter1)
        ax.set_ylabel('Av. Reward')
        ax.grid()
        file = path.join('images', f'rewards_{name}_sweep_{parameter1}_{parameter2}.png')
        plt.savefig(file, dpi=300, bbox_inches="tight")