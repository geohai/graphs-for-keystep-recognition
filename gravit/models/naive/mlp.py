import torch
from torch.nn import Module, ModuleList, Conv1d, Sequential, ReLU, Dropout, Linear


class SimpleMLP(Module):
    def __init__(self, cfg):
        input_dim = cfg['input_dim']
        final_dim = cfg['final_dim']
        hidden_size = 1056
        super(SimpleMLP, self).__init__()
        self.fc1 = Linear(input_dim, hidden_size)  # First fully connected layer
        self.relu = ReLU()                          # ReLU activation function
        self.fc2 = Linear(hidden_size, final_dim)

        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

