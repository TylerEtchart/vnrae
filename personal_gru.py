import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.autograd import Variable

from dataset import Dataset


#
# ===============================================
# ===============================================
# ===============================================
#
# Pytorch Gru


class PyTorchGru(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(PyTorchGru, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.hidden_size = hidden_size

    def forward(self, input_var, hidden):
        output, hidden = self.gru(input_var, hidden)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))


#
# ===============================================
# ===============================================
# ===============================================
#
# My Gru


class PyTorchGru(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(PyTorchGru, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.hidden_size = hidden_size

    def forward(self, input_var, hidden):
        output, hidden = self.gru(input_var, hidden)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))


#
# ===============================================
# ===============================================
# ===============================================
#
# Testing ground


# Vars
INPUT_SIZE = 10
HIDDEN_SIZE = 20
NUM_LAYERS = 1

# Input and output
testing_input_var = Variable(torch.Tensor(np.ones((1, 1, INPUT_SIZE))))
testing_output_var = Variable(torch.Tensor(np.ones((1, 1, HIDDEN_SIZE))))

#
# ------------------------------
# PyTorch Gru

# init gru
pygru = PyTorchGru(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
hidden = pygru.init_hidden()

# Outputs of pygru
output = pygru(testing_input_var, hidden)

print(output)

#
# ------------------------------
# My Gru

pygru = PyTorchGru(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
hidden = pygru.init_hidden()

# Outputs of pygru
output = pygru(testing_input_var, hidden)

print(output)


# criterion = nn.MSELoss()