from environments.games import Triqui
from environments.GridBoard import addTuple, randPair, GridBoard
from agents.MBagents import AlphaBetaAgent
from utils.interaction import EnvfromGameAndPl2
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from pathlib import Path
from os import path
from PIL import Image
import numpy as np
from typing import Optional, Tuple, List, Dict, Union

image_folder = Path.cwd() / Path('environments', 'images')
image_folder.mkdir(parents=True, exist_ok=True)
to_array_folder = Path.cwd() / Path('environments', 'to_array')
to_array_folder.mkdir(parents=True, exist_ok=True)

class ABC():
    
    def __init__(self):
        self.nA = 2
        self.action_space = [0,1]
        self.nS = 3
        self.A = 0
        self.B = 1
        self.C = 2
        self.LEFT = 0
        self.RIGHT = 1
        P = {}
        P[self.A] = {a:[] for a in range(self.nA)}
        P[self.A][self.LEFT] = [(1, self.A, -1, False)]
        P[self.A][self.RIGHT] = [(0.1, self.A, -1, False), (0.9, self.B, -1, False)]
        P[self.B] = {a:[] for a in range(self.nA)}
        P[self.B][self.LEFT] = [(1, self.A, -1, False)]
        P[self.B][self.RIGHT] = [(0.1, self.B, -1, False), (0.9, self.C, 10, True)]
        P[self.C] = {a:[] for a in range(self.nA)}
        self.P = P
        self.dict_acciones = {self.LEFT:'LEFT', self.RIGHT:'RIGHT'}
        self.dict_states = {self.A:'A', self.B:'B', self.C:'C'}
        self.p_right = 0.9
        self.state = self.A
        
    def reset(self):
        self.state = self.A
        return self.state
    
    def step(self, action):
        s = self.state
        p = self.P[s][action]
        indice = np.random.choice(range(len(p)), p=[x[0] for x in p])
        new_state = p[indice][1]
        self.state = new_state
        reward = p[indice][2]
        done = p[indice][3]
        return new_state, reward, done    

    def render(self):
        print(f'Estado: {self.state}')

    def __str__(self):
        string = ''
        for s in range(self.nS):
            string += '\n'+'-'*20
            string += f'\nState: {self.dict_states[s]}'
            for a in range(self.nA):
                string += f'\nAction:{self.dict_acciones[a]}'
                for x in self.P[s][a]:
                    string += f'\n| probability:{x[0]}, '
                    string += f'new_state:{self.dict_states[x[1]]}, '
                    string += f'reward:{x[2]}, '
                    string += f'done?:{x[3]} |'
        return string
    

class TriquiEnv(EnvfromGameAndPl2) :
    '''
    Environment for playing triqui against a minimax player.
    '''

    def __init__(self):
        triqui_base = Triqui()
        player2 = AlphaBetaAgent(game=triqui_base, 
                                 player=2, 
                                 max_lim=100)
        super().__init__(game=triqui_base, other_player=player2)
        self.list_acciones = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]



class GridworldEnv():
    """
    A 4x4 Grid World environment from Sutton's Reinforcement 
    Learning book chapter 4. Termial states are top left and
    bottom right corners. Actions are (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions that going off the edge leave the agent in current state.
    Reward of -1 at each step until agent reaches a terminal state.
    """

    def __init__(self, 
                 shape:Optional[Tuple[int,int]]=(4,4),
                 render_mode:Optional[Union[str, None]]=None):
        assert(shape[0] == shape[1])
        self.shape = shape
        self.nS = np.prod(self.shape)
        self.nA = 4
        self.action_space = list(range(self.nA))
        self.state = np.random.randint(1, self.nS - 2)
        self.NORTH = 0
        self.WEST = 1
        self.SOUTH = 2
        self.EAST = 3
        P = {}
        for s in range(self.nS):
            P[s] = {a: [] for a in range(self.nA)}
            # Per state and action provide list as follows
            # P[state][action] = [(probability, next_state, reward, done)]
            # Assignment is obtained by means of method _transition_prob
            position = self._State2Car(s)
            P[s][self.NORTH] = self._transition_prob(position, [0, 1])
            P[s][self.WEST] = self._transition_prob(position, [-1, 0])
            P[s][self.SOUTH] = self._transition_prob(position, [0, -1])
            P[s][self.EAST] = self._transition_prob(position, [1, 0])
        # We expose the model of the environment for dynamic programming
        # This should not be used in any model-free learning algorithm
        self.P = P
        self.dict_acciones = {0:"⬆", 1:"⬅", 2:"⬇", 3:"➡"}
        self.proportion = 5
        self.render_mode = render_mode

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = np.clip(coord[0], 0, self.shape[0] - 1)
        coord[1] = np.clip(coord[1], 0, self.shape[1] - 1)
        return coord

    def _Car2State(self, casilla:tuple) -> int:
        X, Y = casilla
        return np.ravel_multi_index((Y, X), self.shape)

    def _State2Car(self, index:int) -> tuple:
        Y, X = np.unravel_index(index, self.shape)
        return (X, Y)

    def _transition_prob(self, current, delta):
        """
        Model Transitions. Prob is always 1.0.
        :param current: Current position on the grid as (x, y)
        :param delta: Change in position for transition
        :return: [(1.0, new_state, reward, done)]
        """
        # if stuck in terminal state
        current_state = self._Car2State(current)
        if current_state == self._Car2State((self.shape[0] - 1, 0)) or current_state == self._Car2State((0, self.shape[1] - 1)):
            return [(1.0, current_state, 0, True)]
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = self._Car2State(new_position)
        is_done = new_state == self._Car2State((self.shape[0] - 1, 0)) or new_state == self._Car2State((0, self.shape[1] - 1))
        return [(1.0, new_state, -1, is_done)]

    def reset(self):
        self.state = np.random.randint(1, self.nS - 2)
        return self.state
    
    def step(self, action):
        s = self.state
        p = self.P[s][action]
        indice = np.random.choice(range(len(p)), p=[x[0] for x in p])
        new_state = p[indice][1]
        self.state = new_state
        reward = p[indice][2]
        done = p[indice][3]
        return new_state, reward, done    

    def _find_figsize(self):
        x, y = self.shape
        if x == y:
            return (self.proportion,self.proportion)
        elif x > y:
            return (int(self.proportion*(x/y)),self.proportion)
        else:
            return (self.proportion,int(self.proportion*(y/x)))
        
    def _find_offset(self):
        return 1/(self.shape[0]*2), 1/(self.shape[1]*2)

    def render(self):
        # Dibuja el laberinto
        fig, axes = plt.subplots(figsize=self._find_figsize())
        # Dibujo el tablero
        step_x = 1./self.shape[0]
        step_y = 1./self.shape[1]
        tangulos = []
        # Borde del tablero
        tangulos.append(patches.Rectangle((0,0),0.998,0.998,\
                                            facecolor='xkcd:sky blue',\
                                            edgecolor='black',\
                                            linewidth=1))
        offsetX, offsetY = self._find_offset()
        #Poniendo las salidas
        for casilla in [(0,self.shape[1]-1), (self.shape[0]-1,0)]:
            X, Y = casilla
            file_salida = Path(image_folder, 'salida.png')
            arr_img = plt.imread(file_salida, format='png')
            image_salida = OffsetImage(arr_img, zoom=0.05)
            image_salida.image.axes = axes
            ab = AnnotationBbox(
                image_salida,
                [(X*step_x) + offsetX, (Y*step_y) + offsetY],
                frameon=False)
            axes.add_artist(ab)
		# Creo las líneas del tablero
        for j in range(self.shape[1]):
            # Crea linea horizontal en el rectangulo
            tangulos.append(patches.Rectangle(*[(0, j * step_y), 1, 0.008],\
            facecolor='black'))
        for j in range(self.shape[0]):
            # Crea linea vertical en el rectangulo
            tangulos.append(patches.Rectangle(*[(j * step_x, 0), 0.008, 1],\
            facecolor='black'))
        for t in tangulos:
            axes.add_patch(t)
        #Poniendo agente
        Y, X = np.unravel_index(self.state, self.shape)
        imagen_robot = Path(image_folder, 'robot.png')
        arr_img = plt.imread(imagen_robot, format='png')
        image_robot = OffsetImage(arr_img, zoom=0.125)
        image_robot.image.axes = axes
        ab = AnnotationBbox(
            image_robot,
            [(X*step_x) + offsetX, (Y*step_y) + offsetY],
            frameon=False)
        axes.add_artist(ab)
        axes.axis('off')
        if self.render_mode == 'rgb_array':
            to_array_file = Path(to_array_folder, f'to_array.png')
            plt.savefig(to_array_file)
            return plt.imread(to_array_file)
        else:
            plt.show()
            return axes

    def __str__(self):
        string = ''
        for s in range(self.nS):
            string += '\n'+'-'*20
            string += f'\nState: {s} at {np.unravel_index(s, self.shape)}'
            for a in range(self.nA):
                string += f'\nAction:{self.dict_acciones[a]}'
                for x in self.P[s][a]:
                    string += f'\n| probability:{x[0]}, '
                    Y, X = np.unravel_index(x[1], self.shape)
                    string += f'new_state:{x[1]} at ({X}, {Y}), '
                    string += f'reward:{x[2]}, '
                    string += f'done?:{x[3]} |'
        return string
    

class PitLaberynth:

    def __init__(self, size:Optional[int]=4, 
                 mode:Optional[str]='static', 
                 render_mode:Optional[str]='human'):
        if size >= 4:
            self.board = GridBoard(size=size)
        else:
            print("Minimum board size is 4. Initialized to size 4.")
            self.board = GridBoard(size=4)

        #Add pieces, positions will be updated later
        self.board.addPiece('Player','P',(0,0))
        self.board.addPiece('Goal','+',(1,0))
        self.board.addPiece('Pit','-',(2,0))
        self.board.addPiece('Wall','W',(3,0))

        self.mode = mode
        self.size = size

        if mode == 'static':
            self.initGridStatic()
        elif mode == 'player':
            self.initGridPlayer()
        else:
            self.initGridRand()

        self.dict_acciones = {
            0: 'u',
            1: 'd',
            2: 'l',
            3: 'r',
        }
        self.render_mode = render_mode

    def reset(self):
        self.board = GridBoard(size=self.size)
        self.board.addPiece('Player','P',(0,0))
        self.board.addPiece('Goal','+',(1,0))
        self.board.addPiece('Pit','-',(2,0))
        self.board.addPiece('Wall','W',(3,0))
        if self.mode == 'static':
            self.initGridStatic()
        elif self.mode == 'player':
            self.initGridPlayer()
        else:
            self.initGridRand()
        return self.get_state()

    #Initialize stationary grid, all items are placed deterministically
    def initGridStatic(self):
        #Setup static pieces
        self.board.components['Player'].pos = (0,3) #Row, Column
        self.board.components['Goal'].pos = (0,0)
        self.board.components['Pit'].pos = (0,1)
        self.board.components['Wall'].pos = (1,1)

    #Check if board is initialized appropriately (no overlapping pieces)
    #also remove impossible-to-win boards
    def validateBoard(self):
        valid = True

        player = self.board.components['Player']
        goal = self.board.components['Goal']
        wall = self.board.components['Wall']
        pit = self.board.components['Pit']

        all_positions = [piece for name,piece in self.board.components.items()]
        all_positions = [player.pos, goal.pos, wall.pos, pit.pos]
        if len(all_positions) > len(set(all_positions)):
            return False

        corners = [(0,0),(0,self.board.size), (self.board.size,0), (self.board.size,self.board.size)]
        #if player is in corner, can it move? if goal is in corner, is it blocked?
        if player.pos in corners or goal.pos in corners:
            val_move_pl = [self.validateMove('Player', addpos) for addpos in [(0,1),(1,0),(-1,0),(0,-1)]]
            val_move_go = [self.validateMove('Goal', addpos) for addpos in [(0,1),(1,0),(-1,0),(0,-1)]]
            if 0 not in val_move_pl or 0 not in val_move_go:
                #print(self.display())
                #print("Invalid board. Re-initializing...")
                valid = False

        return valid

    #Initialize player in random location, but keep wall, goal and pit stationary
    def initGridPlayer(self):
        #height x width x depth (number of pieces)
        self.initGridStatic()
        #place player
        self.board.components['Player'].pos = randPair(0,self.board.size)

        if (not self.validateBoard()):
            #print('Invalid grid. Rebuilding..')
            self.initGridPlayer()

    #Initialize grid so that goal, pit, wall, player are all randomly placed
    def initGridRand(self):
        #height x width x depth (number of pieces)
        self.board.components['Player'].pos = randPair(0,self.board.size)
        self.board.components['Goal'].pos = randPair(0,self.board.size)
        self.board.components['Pit'].pos = randPair(0,self.board.size)
        self.board.components['Wall'].pos = randPair(0,self.board.size)

        if (not self.validateBoard()):
            #print('Invalid grid. Rebuilding..')
            self.initGridRand()

    def validateMove(self, piece, addpos=(0,0)):
        outcome = 0 #0 is valid, 1 invalid, 2 lost game
        pit = self.board.components['Pit'].pos
        wall = self.board.components['Wall'].pos
        new_pos = addTuple(self.board.components[piece].pos, addpos)
        if new_pos == wall:
            outcome = 1 #block move, player can't move to wall
        elif max(new_pos) > (self.board.size-1):    #if outside bounds of board
            outcome = 1
        elif min(new_pos) < 0: #if outside bounds
            outcome = 1
        elif new_pos == pit:
            outcome = 2

        return outcome

    def step(self, a):
        action = self.dict_acciones[a]
        #need to determine what object (if any) is in the new grid spot the player is moving to
        #actions in {u,d,l,r}
        def checkMove(addpos):
            if self.validateMove('Player', addpos) in [0,2]:
                new_pos = addTuple(self.board.components['Player'].pos, addpos)
                self.board.movePiece('Player', new_pos)
        if action == 'u': #up
            checkMove((-1,0))
        elif action == 'd': #down
            checkMove((1,0))
        elif action == 'l': #left
            checkMove((0,-1))
        elif action == 'r': #right
            checkMove((0,1))
        else:
            pass
        state = self.get_state()
        reward = self.reward()
        pos_player = self.board.components['Player'].pos
        pos_goal = self.board.components['Goal'].pos
        pos_pit = self.board.components['Pit'].pos
        done = True if (pos_player == pos_goal or pos_player == pos_pit) else False
        return state, reward, done, None
    
    def get_state(self):
        return self.board.render_np()

    def reward(self):
        if (self.board.components['Player'].pos == self.board.components['Pit'].pos):
            return -1000
        elif (self.board.components['Player'].pos == self.board.components['Goal'].pos):
            return 10
        else:
            return -1

    def render(self):
        if self.render_mode is None:
            return
        # Get the board as an array
        board = self.board.render_np()
		# Plot the grid
        fig, axes = plt.subplots(figsize=(6, 6))
        step = 1./self.size
        # offsetX, offsetY = 0.125, 0.125
        offsetX, offsetY = 0.5/self.size, 0.5/self.size
        tangulos = []
        tangulos.append(patches.Rectangle((0,0),0.998,0.998,\
                                        facecolor='cornsilk',\
                                        edgecolor='black',\
                                        linewidth=2))
        for j in range(self.size):
            locacion = j * step
            # Crea linea horizontal en el rectangulo
            tangulos.append(patches.Rectangle(*[(0, locacion), 1, 0.008],\
                    facecolor='black'))
            # Crea linea vertical en el rectangulo
            tangulos.append(patches.Rectangle(*[(locacion, 0), 0.008, 1],\
                    facecolor='black'))
        for t in tangulos:
            axes.add_patch(t)
        # Plot exit
        exit = np.where(board[1] == 1)
        y, x = exit[0][0], exit[1][0]
        y = (self.size - 1) - y
        path_image_exit = Path(image_folder, 'PitLaberynth', 'exit.png')
        arr_img = plt.imread(path_image_exit, format='png')
        image_salida = OffsetImage(arr_img, zoom=0.125/(self.size**0.7))
        image_salida.image.axes = axes
        ab = AnnotationBbox(
            image_salida,
            [(x*step) + offsetX, (y*step) + offsetY],
            frameon=False)
        axes.add_artist(ab)
        # Plot pit
        pit = np.where(board[2] == 1)
        y, x = pit[0][0], pit[1][0]
        y = (self.size - 1) - y
        path_image_pit = Path(image_folder, 'PitLaberynth', 'pit.png')
        arr_img = plt.imread(path_image_pit, format='png')
        image_robot = OffsetImage(arr_img, zoom=0.8/(self.size**0.7))
        image_robot.image.axes = axes
        ab = AnnotationBbox(
            image_robot,
            [(x*step) + offsetX, (y*step) + offsetY],
            frameon=False)
        axes.add_artist(ab)
        # Plot wall
        wall = np.where(board[3] == 1)
        y, x = wall[0][0], wall[1][0]
        y = (self.size - 1) - y
        t = patches.Rectangle(*[(x*step,y*step), step,step], facecolor='black')
        axes.add_patch(t)
        # Plot player
        player = np.where(board[0] == 1)
        y, x = player[0][0], player[1][0]
        y = (self.size - 1) - y
        path_image_robot = Path(image_folder, 'PitLaberynth', 'robot.png')
        arr_img = plt.imread(path_image_robot, format='png')
        image_robot = OffsetImage(arr_img, zoom=0.5/(self.size**0.7))
        image_robot.image.axes = axes
        ab = AnnotationBbox(
            image_robot,
            [(x*step) + offsetX, (y*step) + offsetY],
            frameon=False)
        axes.add_artist(ab)
        # Erase axis
        axes.axis('off')
        if self.render_mode == 'human':           
            plt.show()
        elif self.render_mode == 'rgb_array':
            image_file = Path(to_array_folder, 'rendering.png')
            plt.savefig(image_file)
            return np.asarray(Image.open(image_file))

    def close(self):
        pass