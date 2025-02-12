import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_length: int):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(int(input_length), int(input_length))
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        return self.activation(self.dense_layer(x))