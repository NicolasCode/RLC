'''
© Edgar Andrade 2023
Email: edgar.andrade@urosario.edu.co

-----------------------------------------------
Helper functions to gather, process and visualize data

Includes:
    - Episode, Runs the environment for a number of rounds and keeps tally of everything.
    - Experiment, Compares given models on a number of measures.
    - EnvfromGameAndPl2
-----------------------------------------------
'''

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from copy import deepcopy
from utils.plot_utils import Plot
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from time import sleep
from typing import Union, List, Tuple
from agents.BaseAgent import Agent
from utils.interpreters import id_state
from gymnasium.utils.save_video import save_video

class Episode :
    '''
    Runs the environment for a number of rounds and keeps tally of everything.
    '''

    def __init__(self, 
                 environment:any,
                 env_name:str, 
                 agent:any, 
                 model_name:str, 
                 num_rounds:int, 
                 id:int=0,
                 state_interpreter=id_state) -> None:
        self.environment = environment
        self.env_name = env_name
        self.agent = agent
        self.model_name = model_name
        self.num_rounds = num_rounds
        self.done = False
        self.T = 1
        self.id = id
        self.sleep_time = 0.3
        self._clean_state = state_interpreter
        state_ = self.environment.reset()
        state = self._clean_state(state_)
        self.initial_state = state
        if agent is not None:
            self.agent.restart()
            self.agent.states.append(state)
            # lengths = f'#states:{len(self.agent.states)} -- #actions:{len(self.agent.actions)} -- #rewards:{len(self.agent.rewards)} -- #dones:{len(self.agent.dones)}'
            # print('-->', lengths)

    def play_round(self, verbose:int=0, learn:bool=True) -> None:
        '''
        Plays one round of the game.
        Input:
            - verbose, to print information.
                0: no information
                1: only number of simulation
                2: simulation information
                3: simulation and episode information
                4: simulation, episode and round information
            - learn, a boolean to determine if agent learning is enabled.
        '''
        # Ask agent to make a decision
        try:
            action = self.agent.make_decision()
        except Exception as e:
            raise Exception('Oh oh', e)
        # Update records
        self.agent.actions.append(action)
        # Runs the environment and obtains the next_state, reward, done, info
        result = self.environment.step(action)            
        next_state = self._clean_state(result[0])
        reward = result[1]
        done = result[2]
        # Prints info
        if verbose > 3:
            state = self.agent.states[-1]
            print(f'\tThe state is => {state}')
            print(f'\tAgent takes action => {action}')
            print(f'\tThe state obtained is => {next_state}')
            print(f'\tThe reward obtained is => {reward}')
            print(f'\tEnvironment is finished? => {done}')
        # lengths = f'#states:{len(self.agent.states)} -- #actions:{len(self.agent.actions)} -- #rewards:{len(self.agent.rewards)} -- #dones:{len(self.agent.dones)}'
        # print('=>', lengths)
        # Agent learns
        if learn:
            self.agent.update(next_state, reward, done)
        # Update records
        # lengths = f'#states:{len(self.agent.states)} -- #actions:{len(self.agent.actions)} -- #rewards:{len(self.agent.rewards)} -- #dones:{len(self.agent.dones)}'
        # print('==>', lengths)
        self.agent.states.append(next_state)
        self.agent.rewards.append(reward)
        self.agent.dones.append(done)
        # Update round counter
        self.T += 1
        # Update environment "is-finished?"
        self.done = done
        # lengths = f'#states:{len(self.agent.states)} -- #actions:{len(self.agent.actions)} -- #rewards:{len(self.agent.rewards)} -- #dones:{len(self.agent.dones)}'
        # print('===>', lengths)

    def run(self, verbose:int=0, learn:bool=True) -> pd.DataFrame:
        '''
        Plays the specified number of rounds.
        Input:
            - verbose, to print information.
                0: no information
                1: only number of simulation
                2: simulation information
                3: simulation and episode information
                4: simulation, episode and round information
            - learn, a boolean to determine if agent learning is enabled.
        '''
        for round in range(self.num_rounds):
            if not self.done:
                if verbose > 2:
                    print('\n' + '-'*10 + f'Round {round}' + '-'*10 + '\n')
                self.play_round(verbose=verbose, learn=learn)                
            else:
                break
        return self.to_pandas()

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
                    * environment: the environment's name
        '''
        # Include las item in actions list
        self.agent.actions.append(np.nan)
        # Perform cleansing operations on registers
        states, actions, rewards, dones = self.cleanse_registers(self.agent.states, 
                                                                 self.agent.actions, 
                                                                 self.agent.rewards, 
                                                                 self.agent.dones)
        data = {}
        data["episode"] = []
        data["round"] = []
        data["state"] = []
        data["action"] = []
        data["reward"] = []
        data["done"] = []
        for r in range(self.T):
            data["episode"].append(self.id)
            data["round"].append(r)
            data["state"].append(states[r])
            data["action"].append(actions[r])
            data["reward"].append(rewards[r])
            data["done"].append(dones[r])
        df = pd.DataFrame.from_dict(data)        
        df["model"] = self.model_name
        df["environment"] = self.env_name
        return df

    def cleanse_registers(self,
                               states:List[any],
                               actions:List[int],
                               rewards:List[float],
                               dones:List[bool]) -> Tuple[List, List, List, List]:
        lengths = f'#states:{len(states)} -- #actions:{len(actions)} -- #rewards:{len(rewards)} -- #dones:{len(dones)}'
        n = len(states)
        assert(len(actions) == n), lengths
        assert(len(rewards) == n), lengths
        assert(len(dones) == n), lengths
        # Get indices from all nan in dones
        nan_dones = [i for i, d in enumerate(dones) if np.isnan(d)]
        # Check whether more than one episode lies in registers
        num_nan_dones = sum([1 for d in self.agent.dones if np.isnan(d)])
        if num_nan_dones > 1:
            # print('More than one episode in registers!')
            pass
        last_batch_init_index = nan_dones[-1]
        states = states[last_batch_init_index:]
        actions = actions[last_batch_init_index:]
        rewards = rewards[last_batch_init_index:]
        dones = dones[last_batch_init_index:]        
        assert(self.T == len(states)), f'T:{self.T}; len(dones):{len(dones)}\n{nan_dones}\n{dones}\n{self.agent.dones}'
        return states, actions, rewards, dones

    def reset(self) -> None:
        '''
        Reset the episode. This entails:
            reset the environment
            restart the agent 
        '''
        state = self.environment.reset()
        self.T = 1
        self.done = False
        state = self._clean_state(state)
        self.agent.restart()        
        if len(self.agent.states) == 1:
            self.agent.states = [state]
        else:
            self.agent.states.append(state)
        self.T = 1
        self.done = False

    def renderize(self, to_video:bool=False, file:str=None) -> None:
        '''
        Plays the specified number of rounds.
        '''
        if to_video:
            assert(file is not None), 'A folder name must be provided with the argument file='
            rm = self.environment.render_mode
            assert(rm == 'rgb_array'), f'To create video, environment render mode should be rgb_array, not {rm}'
            frame_list = []
        # Initialize img object
        img = plt.imshow(np.array([[0, 0], [0, 0]]))
        for round in range(self.num_rounds):
            if not self.done:
                im = self.environment.render()
                if isinstance(im, np.ndarray):
                    if to_video:
                        frame_list.append(im)
                clear_output(wait=True)
                plt.axis('off')
                img.set_data(im)
                display(plt.gcf())
                sleep(self.sleep_time)
                self.play_round(verbose=0, learn=False)                
            else:
                # clear_output(wait=True)
                # im = self.environment.render()
                # if isinstance(im, np.ndarray):
                #     if to_video:
                #         frame_list.append(im)
                # clear_output(wait=True)
                # plt.axis('off')
                # img.set_data(im)
                # display(plt.gcf())
                break
        if to_video:
            assert(len(frame_list) > 0), 'No frames saved. Check env.render() is providing np.ndarrays.'
            save_video(
                frames=frame_list,
                video_folder=file,
                fps=1/self.sleep_time
            )
   
    def simulate(self, 
                 num_episodes:int=1, 
                 file:str=None, 
                 verbose:int=0, 
                 learn:bool=True) -> pd.DataFrame:
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
            - learn, a boolean to determine if agent learning is enabled.
        Output:
            - Pandas dataframe with the following variables:

                Variables:
                    * id_sim: a unique identifier for the simulation
                    * round: the round number
                    * action: the player's action
                    * reward: the player's reward
                    * done: whether the environment is active or not
                    * model: the model's name
                    * environment: the environment's name
        '''
        # Create the list of dataframes
        data_frames = []
        # Run the number of episodes
        for ep in tqdm(range(num_episodes)):
            if verbose > 1:
                print('\n' + '='*10 + f'Episode {ep}' + '='*10 + '\n')
            # Reset the episode
            # lengths = f'#states:{len(self.agent.states)} -- #actions:{len(self.agent.actions)} -- #rewards:{len(self.agent.rewards)} -- #dones:{len(self.agent.dones)}'
            # print('//>', lengths)
            self.reset()
            self.id = ep
            # Run the episode
            # lengths = f'#states:{len(self.agent.states)} -- #actions:{len(self.agent.actions)} -- #rewards:{len(self.agent.rewards)} -- #dones:{len(self.agent.dones)}'
            # print('///>', lengths)
            df = self.run(verbose=verbose, learn=learn)
            # Include episode in list of dataframes
            data_frames.append(df)
        # Concatenate dataframes
        data = pd.concat(data_frames, ignore_index=True)
        if file is not None:
            data.to_csv(file)
        return data
    

class Experiment :
    '''
    Compares given models on a number of measures.
    '''

    def __init__(self, \
                 environment:any, \
                 env_name:str, \
                 num_rounds:int, \
                 num_episodes:int, \
                 num_simulations:int, \
                 state_interpreter=id_state):
        '''
        Input:
            - environment, object with the environment on which to test the agents.
            - env_name, the environment's name.
            - num_rounds, int with the number of rounds.
            - num_episodes, int with the number of episodes.
            - num_simulations, int with the number of times the environment should be
                restarted and run the episodes again.
        '''
        self.environment = environment
        self.env_name = env_name
        self.num_rounds = num_rounds
        self.num_episodes = num_episodes
        self.num_simulations = num_simulations
        self.state_interpreter = state_interpreter
        self.data = None

    def run_experiment(self, \
                       agents:List[any], \
                       names:List[str], \
                       measures:List[str], \
                       learn:bool=True) -> None:
        '''
        For each agent, runs the simulation the stipulated number of times,
        obtains the data and shows the plots on the given measures.
        Input:
            - agents, list of agent objects.
            - names, list of names of the models implemented by the agents.
            - measures, list of measures, which could contain the following strings:
                * 'reward'
                * 'round_reward'
                * 'hist_reward'
            - learn, a boolean to enable learning.
        '''
        # Creates the list of dataframes
        data_frames = []
        # Run simulations
        for k in tqdm(range(self.num_simulations), desc='Running simulations'):
            # Reset all agents
            if learn:
                for agent in agents:
                    agent.reset()
            # Iterate over episodes
            for ep in tqdm(range(self.num_episodes), desc='\tRunning episodes', leave=False):
                # Initialize Episode
                sim_core = Episode(environment=self.environment, \
                                   env_name=self.env_name, \
                                   agent=None, \
                                   model_name=None,\
                                   num_rounds=self.num_rounds,\
                                   state_interpreter=self.state_interpreter)
                # Keep a unique id number for the episode
                sim_core.id = ep
                counter_agent = -1
                for agent in agents:
                    counter_agent += 1
                    # Copy Episode and place agent
                    sim = deepcopy(sim_core)
                    # Restart agent for a new episode
                    agent.restart()
                    sim.agent = agent
                    sim.agent.states.append(sim.initial_state)
                    sim.model_name = names[counter_agent]
                    # Run episode over agent
                    df = sim.run(verbose=False, learn=learn)
                    df["simulation"] = k
                    df["model"] = names[counter_agent]
                    data_frames.append(df)
        # Consolidate data
        data = pd.concat(data_frames, ignore_index=True)
        self.data = data
        # Create plots
        for m in measures:
            if m == 'reward':
                ax = Plot(data).plot_rewards(m)
            if m == 'round_reward':
                ax = Plot(data).plot_round_reward(m)
            if m == 'hist_reward':
                ax = Plot(data).plot_histogram_rewards(m)
            try:
                ax.set_title(m)
            except:
                pass
            plt.show()
        return agents

    def run_sweep1(self, \
                    agent:any, \
                    name:str, \
                    parameter:str, \
                    values:List[Union[int, float, str]], \
                    measures:List[str], \
                    learn:bool=True) -> None:
        '''
        For each agent, runs a parameter sweep the stipulated number
        of times, obtains the data and shows the plots on the given measures.
        Input:
            - agent, an object agent.
            - name, the name of the model implemented by the agent.
            - parameter, a string with the name of the parameter.
            - values, a list with the parameter's values.
            - measures, list of measures, which could contain the following strings:
                * 'reward'
                * 'round_reward'
            - learn, a boolean to enable learning.
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
        # Run the experiment
        self.run_experiment(agents=agents,\
                            names=names,\
                            measures=[],\
                            learn=learn)            


class EnvfromGameAndPl2:
    '''
    Implementa un entorno a partir de un juego y del segundo jugador.
    '''
    def __init__(self, game:any, other_player:any):
        self.other_player = other_player
        self.initial_game = deepcopy(game)
        self.game = game
        self.state = self.game.estado_inicial
        self.list_acciones = None

    def reset(self):
        self.game = deepcopy(self.initial_game)
        self.state = self.game.estado_inicial
        self.other_player.reset()
        self.other_player.states.append(self.state)
        return self.state

    def render(self):
        self.game.render(self.state)

    def test_objetivo(self, state):
        if not self.game.es_terminal(state):
            return False
        else:
            player = self.game.player(state)
            return self.utilidad(state, player) > 0
        
    def acciones_aplicables(self, state):
        return self.game.acciones(state)

    def step(self, action):
        if self.list_acciones is not None:
            action_ = self.list_acciones[action] 
        else:
            action_ = action
        state = self.state
        playing = self.game.player(state)
        # print(f'player {playing} in state {state} makes move {action}')
        # First player made a move. Get new state, reward, done
        try:
            new_state = self.game.resultado(state, action_)
        except Exception as e:
            if action_ not in self.game.acciones(state):
                # Punish agent for playing an impossible action
                return state, -1000, True
            print(state)
            raise Exception(e)
        # self.game.render(new_state)
        # print(f'obtains {new_state}')
        reward = self.game.utilidad(new_state, playing)
        reward = reward if reward is not None else 0
        done = self.game.es_terminal(new_state)
        # If not done, second player makes a move
        if not done:
            playing = self.game.player(new_state)
            # Actualize second player with previous move
            self.other_player.states.append(new_state)
            if hasattr(self.other_player, 'choices'):
                possible_actions = self.game.acciones(new_state)
                self.other_player.choices = possible_actions
            # Second player makes a move
            other_player_action = self.other_player.make_decision()
            if self.other_player.debug:
                print(f'Negras mueven en {other_player_action}')
            # print(f'player {playing} in state {new_state} makes move {other_player_action}')
            # Get new state, reward, done
            new_state = self.game.resultado(new_state, other_player_action)
            # print(f'obtains {new_state}')
            reward = self.game.utilidad(new_state, playing)
            reward = reward if reward is not None else 0
            done = self.game.es_terminal(new_state)
        # Bookkeeping
        self.state = new_state
        return new_state, reward, done   
    
