import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.autograd import Variable

from dataset import Dataset
from tqdm import tqdm


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


#
# ===============================================
# ===============================================
# ===============================================
#
# My Gru


class MyGru(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MyGru, self).__init__()
        
        self.r_t1 = nn.Linear(input_size, hidden_size)
        self.r_t2 = nn.Linear(hidden_size, hidden_size)
        self.r_t_sig = nn.Sigmoid()
        
        self.z_t1 = nn.Linear(input_size, hidden_size)
        self.z_t2 = nn.Linear(hidden_size, hidden_size)
        self.z_t_sig = nn.Sigmoid()

        self.n_t1 = nn.Linear(input_size, hidden_size)
        self.n_t_tanh = nn.Tanh()

    def forward(self, input_var, hidden):   
        r_t = self.r_t_sig(self.r_t1(input_var) + self.r_t2(hidden))
        z_t = self.z_t_sig(self.z_t1(input_var) + self.z_t2(hidden))

        n_t = self.n_t_tanh(self.n_t1(input_var) + r_t)
        h_t = ((1 - z_t) * n_t) + (z_t * hidden)

        return n_t, h_t


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
LEARNING_RATE = .001

# Input and output
testing_input_var = Variable(torch.Tensor(np.ones((1, 1, INPUT_SIZE))))
testing_output_var = Variable(torch.Tensor(np.ones((1, 1, HIDDEN_SIZE))))
testing_hidden = Variable(torch.zeros(1, 1, HIDDEN_SIZE))

criterion = nn.MSELoss()

#
# ------------------------------
# PyTorch Gru

# init gru
pygru = PyTorchGru(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
pygru_optimizer = optim.SGD(pygru.parameters(), lr=LEARNING_RATE)

# Outputs of pygru
for i in tqdm(range(3000)):
    pygru_optimizer.zero_grad()
    py_output, hidden = pygru(testing_input_var, testing_hidden)
    loss = criterion(py_output, testing_output_var)
    loss.backward()
    pygru_optimizer.step()

print("\nPyGru diff:", np.sum(py_output.data.numpy() - testing_output_var.data.numpy()))
print("Shape:", py_output.shape)
print()

#
# ------------------------------
# My Gru

# init gru
mygru = MyGru(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
mygru_optimizer = optim.SGD(mygru.parameters(), lr=LEARNING_RATE)

# Outputs of pygru
for i in tqdm(range(3000)):
    mygru_optimizer.zero_grad()
    my_output, hidden = mygru(testing_input_var, testing_hidden)
    loss = criterion(my_output, testing_output_var)
    loss.backward()
    mygru_optimizer.step()

print("\nMyGru diff:", np.sum(my_output.data.numpy() - testing_output_var.data.numpy()))
print("Shape:", my_output.shape)

print("\nFinal Diff:", np.sum(py_output.data.numpy() - my_output.data.numpy()))