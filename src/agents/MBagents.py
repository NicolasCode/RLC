'''
© Edgar Andrade 2018
Email: edgar.andrade@urosario.edu.co

-----------------------------------------------
Model-based agent implementing game strategies.

Includes:
    - AlphaBetaAgent, a generic minimax with alpha-beta prunning agent for a zero-sum two-player game.
-----------------------------------------------
'''
from typing import Dict
import numpy as np

class Agent :
    '''
    Defines the basic methods for the agent.
    '''

    def __init__(self):
        self.plan = []
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [False]
        self.turn = 0

    def make_decision(self):
        '''
        Agent makes a decision according to its model.
        '''
        # Chequeamos si ya el agente no tiene un plan (lista de acciones)
        if len(self.plan) == 0:
            # Usamos el programa para crear un plan
            self.program()
        try:
            # La acción a realizar es la primera del plan
            action = self.plan.pop(0)
        except:
            # ¡No hay plan!
            state = self.states[-1]
            raise Exception(f'¡Plan vacío! Revisar reglas en estado {state}')
        self.turn += 1
        return action

    def program(self):
        '''
        Debe ser modificada por cada subclase
        '''
        pass

    def reset(self):
        self.restart()

    def restart(self):
        '''
        Restarts the agent for a new trial.
        '''
        self.plan = []
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [False]
        self.turn = 0


class AlphaBetaAgent(Agent):
    '''
    Generic minimax with alpha-beta prunning agent for a zero-sum two-player game.
    '''
    def __init__(self, game:any, player:str, max_lim:int=2):
        super().__init__()
        self.game = game
        assert(player == 1 or player == 2), f'Player must be either 1 or 2, not {player}'
        self.player = player
        self.max_lim = max_lim
        self.debug = False

    def program(self):
        state = self.states[-1]
        v, action = self._H_minimax_alfa_beta(state, 0, -np.infty, np.infty)
        # print(f'Action {action} has value {v}')
        self.plan.append(action)

    def _H_minimax_alfa_beta(self, state, d, alfa, beta):
        player = self.game.player(state)
        if self.debug:
            print('\n')
            print('='*10)
            print('\t'*d + player)
            print(f'Profundidad:{d}, ¿cutoff?:{self._is_cutoff(state, d)} ({d}>={self.max_lim}?)')
            print('='*10)
        if self._is_cutoff(state, d):
            return self.evaluate(state), None
        elif player == 1:
            # print('Agent 1 is making a decision...')
            v = -np.infty
            for a in self.game.acciones(state):
                board_resultado = self.game.resultado(state, a)
                v2, a2 = self._H_minimax_alfa_beta(board_resultado, d+1, alfa, beta)
                if self.debug:
                    print(a, v2)
                if v2 > v:
                    v = v2
                    accion = a
                    alfa = max(alfa, v)
                if v >= beta:
                    if self.debug:
                        print('prunning beta...', a)
                    return v, accion
            return v, accion
        elif player == 2:
            # print('Agent 2 is making a decision...')
            v = np.infty
            for a in self.game.acciones(state):
                board_resultado = self.game.resultado(state, a)
                v2, a2 = self._H_minimax_alfa_beta(board_resultado, d+1, alfa, beta)
                if self.debug:
                    print(a, v2)
                if v2 < v:
                    v = v2
                    accion = a
                    beta = min(beta, v)
                if v <= alfa:
                    if self.debug:
                        print('prunning alfa...', a)
                    return v, accion
            return v, accion
        else:
            raise NameError(f"Oops! player={player}")  
               
    def evaluate(self, state):
        player_ = self.game.player(state)
        if self.debug:
            print('Juega --->', player_)
        player = self._other_player(player_)
        if self.game.es_terminal(state):
            return self.game.utilidad(state, player)
        if player == 1:
            evaluacion = self.player1_evaluation(state)
            pass
        elif player == 2:
            evaluacion = self.player2_evaluation(state)
            pass
        else:
            raise Exception(f'Oops! player={player}')
        if self.debug:
            print('evaluador', player, '->', evaluacion)
        return evaluacion 
    
    def player1_evaluation(self, state):
        '''
        To be defined by superclass
        '''
        pass
   
    def player2_evaluation(self, state):
        '''
        To be defined by superclass
        '''
        pass

    def _other_player(self, player):
        return 1 if player == 2 else 2

    def _is_cutoff(self, state, d):
        if self.game.es_terminal(state):
            return True
        # Considering d a ply, so a round is two plies (2*d).
        # But only evaluate an odd number of plies downwards.
        elif 2*(d-1) + 1 >= self.max_lim: 
            return True
        else:
            return False