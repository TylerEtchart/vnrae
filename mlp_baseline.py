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
# Helper Functions


def pad_sequence(sequence, pad_len):
    seq_len = len(sequence)
    seq_len = seq_len if seq_len < pad_len else pad_len

    padded_seq = np.zeros((pad_len))
    padded_seq[:seq_len] = sequence[:seq_len]
    return padded_seq.astype(int)


def match_sequences(sequence1, sequence2, match_seq1=True):
    if match_seq1:
        pad_len = len(sequence1)
        seq_len1 = pad_len
        seq_len2 = len(sequence2)
        seq_len2 = seq_len2 if seq_len2 < pad_len else pad_len
    else:
        pad_len = len(sequence2)
        seq_len2 = pad_len
        seq_len1 = len(sequence1)
        seq_len1 = seq_len1 if seq_len1 < pad_len else pad_len
    

    padded_seq1 = np.zeros((pad_len))
    padded_seq2 = np.zeros((pad_len))

    padded_seq1[:seq_len1] = sequence1[:seq_len1]
    padded_seq2[:seq_len2] = sequence2[:seq_len2]

    return padded_seq1.astype(int), padded_seq2.astype(int)


#
# ===============================================
# ===============================================
# ===============================================
#
# MLP


class MlpBaseline(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MlpBaseline, self).__init__()
        l_in = nn.Linear(input_dim, hidden_dim)
        relu = nn.ReLU()
        l_out = nn.Linear(hidden_dim, output_dim)
        softmax = nn.LogSoftmax(dim=-1)
        self.mlp = nn.Sequential(l_in, relu, l_out, softmax)

    def forward(self, inputs):
        return self.mlp(inputs)


#
# ===============================================
# ===============================================
# ===============================================
#
# Define vars


# make the dataset
dataset = Dataset()

# define constants
INPUT_DIM = dataset.vocab_size
HIDDEN_DIM = 200
OUTPUT_DIM = dataset.vocab_size
LEARNING_RATE = .001
PAD_LENGTH = 200
PAD = False

# make the MLP
mlp = MlpBaseline(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

# define the loss
criterion = nn.NLLLoss()

# define the optimizer
optimizer = optim.SGD(mlp.parameters(), lr=LEARNING_RATE)


#
# ===============================================
# ===============================================
# ===============================================
#
# Training loop


for epoch in range(30):
    for convo_i in range(dataset.size()):
        # zero out gradient
        optimizer.zero_grad()

        # get data
        x_original, y_original = dataset.next_batch()

        # pad sequences
        if PAD:
            x_padded = pad_sequence(x_original, PAD_LENGTH)
            y_padded = pad_sequence(y_original, PAD_LENGTH)
        else:
            x_padded, y_padded = match_sequences(x_original, y_original, match_seq1=True)

        # convert to onehots
        x = dataset.to_onehot(x_padded, long_type=False)
        y = Variable(torch.LongTensor(y_padded))
        # y = dataset.to_onehot(y, long_type=True)

        # run through mlp
        output = mlp(x)

        # collect loss
        loss = criterion(output, y)

        # update mlp
        loss.backward()
        optimizer.step()

        # output loss
        if convo_i % 10 == 0:
            print("Epoch: {}, Conversation: {}, Loss: {}".format(epoch, convo_i, loss.data.numpy()[0]))

        # output example
        if convo_i % 100 == 0:
            outs = output.data.numpy()
            outs = np.argmax(outs, axis=1)
            print("\n-------------------------")
            print("Offer:")
            print("".join([dataset.chars[x_padded[i]] for i in range(len(x_padded))]))
            print("Target:")
            print("".join([dataset.chars[y_padded[i]] for i in range(len(y_padded))]))
            print("Output:")
            print("".join([dataset.chars[outs[i]] for i in range(len(outs))]))
            print("-------------------------\n")