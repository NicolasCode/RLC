import torch

'''Class with NN arquitectures'''


class FFN(torch.nn.Module):
    '''
    A 3-layer Perceptron
    '''
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        """
        Args:
            input_size (int): size of the input vectors
            hidden_size (int): the output size of the first Linear layer
            output_size (int): the output size of the second Linear layer
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # -------------------------------------
        # Defining the layers
        # -------------------------------------
        # Hidden layer
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.tanh = torch.nn.Tanh()
        # Output layer
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x_in):
        """The forward pass of the network        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, input_dim)
        Returns:
            the resulting tensor. tensor.shape should be (batch, vocabulary_size)
        """
        # Run the first layer 
        out = self.fc1(x_in)
        out = self.tanh(out)
        # Propagate to output layer
        out = self.fc2(out)
        return out
    

class FFN_D(torch.nn.Module):
    '''
    A 4-layer Perceptron
    '''
    def __init__(self,\
                 input_size:int, 
                 hidden_size_1:int, 
                 hidden_size_2:int, 
                 hidden_size_3:int, 
                 output_size:int):
        """
        Args:
            input_size (int): size of the input vectors
            hidden_size (int): the output size of the first Linear layer
            output_size (int): the output size of the second Linear layer
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.output_size = output_size
        # -------------------------------------
        # Defining the layers
        # -------------------------------------
        # First hidden layer
        self.fc1 = torch.nn.Linear(input_size, hidden_size_1)
        self.tanh = torch.nn.Tanh()
        # Second hidden layer
        self.fc2 = torch.nn.Linear(input_size, hidden_size_2)
        self.relu = torch.nn.ReLU()
        # Third hidden layer
        self.fc3 = torch.nn.Linear(input_size, hidden_size_3)
        self.relu = torch.nn.ReLU()
        # Output layer
        self.fc4 = torch.nn.Linear(hidden_size_3, output_size)

    def forward(self, x_in):
        """The forward pass of the network        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, input_dim)
        Returns:
            the resulting tensor. tensor.shape should be (batch, vocabulary_size)
        """
        # Run the first layer 
        out = self.fc1(x_in)
        out = self.tanh(out)
        # Propagate to second layer 
        out = self.fc2(x_in)
        out = self.relu(out)
        # Propagate to second layer 
        out = self.fc3(x_in)
        out = self.relu(out)
        # Propagate to output layer
        out = self.fc4(out)
        return out
    

''' Convolutional network '''
class CNN_CarRacing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = torch.nn.MaxPool2d(stride=2, kernel_size=2)
        
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = torch.nn.MaxPool2d(stride=2, kernel_size=2)
        
        self.fc1 = torch.nn.Linear( 11520 , 2024)
        self.fc2 = torch.nn.Linear( 2024 , 5 )

    def forward(self, x_in):
        out = self.conv1(x_in)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.maxpool2(out)

        out = torch.flatten(out)

        out = self.fc1(out)
        out = self.fc2(out)

        return out

''' 
# Create an instance of the CNN model
model = CNN()

# Dummy input tensor with shape (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 96, 96)

# Forward pass
output = model(input_tensor)

print("Input shape:", input_tensor.shape)
print("Output shape:", output.shape)
'''