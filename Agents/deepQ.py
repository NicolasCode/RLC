'''
Classes for implementing the Q functions with neural networks
'''
import torch
import numpy as np
from os import path
from Agents.networks import FFN, FFN_D, CNN_CarRacing

class Uniform_testQ():
    '''
    Defines a Q function that returns 0 for all state, action pairs
    '''
    def __init__(self, parameters):
        self.action = 0
        self.nA= parameters["nA"]

    def model(self, state):
        return torch.Tensor([1 if i == self.action else 0 for i in range(self.nA)]).unsqueeze(axis=0)

    def predict(self, state, action):
        return 3
    
    def learn(self, state, action, update, alpha):
        pass

    def reset(self):
        pass

class NN_as_Q():
    '''
    Defines a Feedforward Network implementing a Q function
    '''
    def __init__(self, parameters, model):
        self.parameters = parameters
        alpha = parameters["alpha"]
        self.loss_fn = torch.nn.MSELoss()
        self.losses = []
        self.model_file = path.join('data', 'ffn.pt')
        self.model_file_trained = path.join('data', 'ffn_trained.pt')
        self.model = model
        # Save model for when reset is executed
        torch.save(self.model.state_dict(), self.model_file)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
       
    def predict(self, state, action, noise=False):
        with torch.no_grad():
            # Gets predicted Q values
            Qs = self.model(torch.from_numpy(state).float())
            # Transforms to list
            if len(Qs.shape) > 1:
                Qs = Qs[0]

            Qs = Qs.data.numpy()  
        return Qs[action]
    
    def learn(self, state, action, update, alpha):
        # Gets the predicted values by the network
        state_ = torch.from_numpy(state)
        qval = self.model(state_.float())
        # print('=>', qval, qval.shape)
        # Check if action is batch
        if isinstance(action, list):
            # Confirm the number of actions
            nA = self.parameters["output_size"]
            # Get the indices for the actions
            mask = torch.tensor([i*nA + a for i, a in enumerate(action)], dtype=torch.long)
            # print('==>', mask)
            # Keep only the q values corresponding to the actions
            X = torch.take(qval, mask)
        else:
            # Keeps only the q value corresponding to the action
            X = qval.squeeze()[action]
        # Adapts the update
        if isinstance(update, list):
            Y = torch.Tensor(update).detach()
        else:
            Y = torch.Tensor([update]).squeeze().detach()
        # print('-->', qval, action, X, Y)
        # Determines the loss
        loss = self.loss_fn(X, Y)
        self.losses.append(loss.item())
        # Clears the gradient
        self.optimizer.zero_grad()
        # Finds the gradients by backward propagation
        loss.backward()
        # Updates the weights with the optimizer
        self.optimizer.step()
        # print('--', self.predict(state, action), '--')

    def reset(self):
        # Gets the model from the file
        self.model.load_state_dict(torch.load(self.model_file))
        # Resets the losses
        self.losses = []


class FFNQ(NN_as_Q):
    '''
    Defines a Feedforward Network implementing a Q function
    '''
    def __init__(self, parameters):
        model = FFN(\
            input_size=parameters["input_size"],
            hidden_size=parameters["hidden_size"],
            output_size=parameters["output_size"]
            )
        super().__init__(parameters, model)
       


class FFNQ_D(NN_as_Q):
    '''
    Defines a 3 layer FFN implementing a Q function
    '''
    def __init__(self, parameters):
        model = FFN_D(\
            input_size=parameters["input_size"],
            hidden_size_1=parameters["hidden_size_1"],
            hidden_size_2=parameters["hidden_size_2"],
            hidden_size_3=parameters["hidden_size_3"],
            output_size=parameters["output_size"]
            )
        super().__init__(parameters, model)


class CNN(NN_as_Q):
    '''
    Defines a convolutional Network implementing a Q function
    '''
    def __init__(self, parameters):
        model = CNN_CarRacing()
        super().__init__(parameters, model)

    def save(self):
        torch.save(self.model.state_dict(), self.model_file_trained)

    def load(self):
        self.model.load_state_dict(torch.load(self.model_file_trained))
        self.model.eval()