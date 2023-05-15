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

class Car(Agent):

    # dist = num in range(0,1) obtained from the sensors
    def __init__(self, distf, distl, distr):
        # initial values
        super().__init__()
        self.angle = 0 
        self.range_angle = 369
        self.speed = 0

        # sensor values
        self.distf = distf
        self.distr = distr
        self.distl = distl
        
        # Opcional
        self.pos = (0,0)

    def restart(self):
        super().restart()
        self.angle = 0
        self.pos = (0,0)
        self.speed = 0

    # intensity is a number in range (1,5)
    def gas(self, intensity):
        self.speed += intensity
    
    # decrement the speed in a 30%, if speed is lower than 5 then set speed in 0
    def brake(self):
        if self.speed > 5:
            self.speed -= self.speed* 0.35
        else:
            self.speed = 0
        
    # intensity is a number in range (1,5)
    def turn_left(self, intensity):
        self.angle = (self.angle + intensity) % self.range_angle
    
    # intensity is a number in range (1,5)
    def turn_right(self, intensity):
        self.angle = (self.angle - intensity) % self.range_angle
        

    def make_decision(self):
        
        """ 
        Nicolas:
        Agregar las acciones al plan en el orden en el que se debe ejecutar, pensar en que
        si tengo un obstaculo a la derecha no solo debo girar, tambien acelerar,
        
        Prof. Edgar:
        ¿ Debo yo definir estas acciones o dejar que las defina la red ? 
        ¿ Debo crear la red bayesiana de el agente ?
        ¿ Son estas funciones suficientes para el funcionamiento de mi agente?
        ¿ Como  puedo testear mi agente en el entorno de Gymnasius? 
        
        """


        if self.distf > 0.5:
            self.brake()
        
        if self.distr > 0.5:
            self.turn_left(4)

        if self.distl > 0.5:
            self.turn_right(4)
        
        else: 
            self.gas(2)

        return super().make_decision()