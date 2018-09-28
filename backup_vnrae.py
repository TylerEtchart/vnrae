import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.optim as optim
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.util import ng_zeros, ng_ones

import fasttext
from dataset_fasttext import Dataset
import sys

#
# ===============================================
# ===============================================
# ===============================================
#
# Globals


# init dataset
dataset = Dataset()

# Sizes
VOCAB_SIZE = 50002
ENCODER_HIDDEN_SIZE = 300 
DECODER_HIDDEN_SIZE = 600
Z_DIMENSION =300 
LEARNING_RATE = .0001
MAX_LENGTH = 300
NUM_LAYERS_FOR_RNNS = 1
CONTEXT_LENGTH = 1

USE_CUDA = True
TEACHER_FORCING = False


#
# ===============================================
# ===============================================
# ===============================================
#
# RNNs

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, ftext):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fc = nn.Linear(input_size, num_layers*hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        
        self.fc.weight.data.copy_(torch.FloatTensor(ftext.vectors).view_as(self.fc.weight.data))
        #self.fc.weight.requires_grad = False


    def forward(self, input_var, hidden):
        if USE_CUDA:
            input_var = input_var.cuda()
        embedded = self.fc(input_var).view(self.num_layers, 1, -1)
        output, hidden = self.rnn(embedded, hidden)

        if type(self.fc.weight.grad) == type(None):
            print("EncoderRNN fc gradiants are none")

        if type(self.rnn.weight_ih_l0.grad) == type(None):
            print("EncoderRNN IH gradiants are none")
        if type(self.rnn.weight_hh_l0.grad) == type(None):
            print("EncoderRNN HH gradiants are none")

        return output, hidden

    def init_hidden_lstm(self):
        result = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                  Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))
        if USE_CUDA:
            return result.cuda()
        else:
            return result

    def init_hidden_gru(self):
        result = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Linear(output_size, num_layers*hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_var, hidden):
        if USE_CUDA:
            input_var = input_var.cuda()

        hidden = hidden.view(self.num_layers, 1, self.hidden_size)

        output = self.embedding(input_var).view(self.num_layers, 1, -1)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))

        if type(self.embedding.weight.grad) == type(None):
            print("DecoderRNN embedding weights are none")
        if type(self.rnn.weight_ih_l0.grad) == type(None):
            print("DecoderRNN IH weights are none")
        if type(self.rnn.weight_hh_l0.grad) == type(None):
            print("DecoderRNN HH weights are none")
        # else:
        #     print("DecoderRNN HH weight sum ", torch.sum(self.rnn.weight_hh_l0.data))

        return output, hidden

    def init_hidden_lstm(self):
        result = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                  Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))
        if USE_CUDA:
            return result.cuda()
        else:
            return result

    def init_hidden_gru(self):
        result = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result


#
# ===============================================
# ===============================================
# ===============================================
#
# Dense Layers


class EncoderDense(nn.Module):
    def __init__(self, hidden_dim, z_dim):
        super(EncoderDense, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_sig = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        x = x.view(-1, self.hidden_dim)
        z_mu = self.fc_mu(x)
        z_sigma = torch.exp(self.fc_sig(x))

        if type(self.fc_mu.weight.grad) == type(None):
            print("EncoderDense mu grad is none")
        if type(self.fc_sig.weight.grad) == type(None):
            print("EncoderDense sig grad is none")

        return z_mu, z_sigma


class DecoderDense(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(DecoderDense, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(z_dim, hidden_dim)

    def forward(self, z):
        hidden = self.fc(z)
        
        if type(self.fc.weight.grad) == type(None):
            print("DecoderDense grad is none")
        # else:
        #     print("DecoderDense weight sum", torch.sum(self.fc.weight.data))

        return hidden


#
# ===============================================
# ===============================================
# ===============================================
#
# VRAE


class VRAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self,
                 dataset,
                 vocab_dim,
                 encoder_hidden_dim,
                 z_dim,
                 decoder_hidden_dim,
                 max_length,
                 num_layers_for_rnns,
                 use_cuda=False):
        super(VRAE, self).__init__()

        #fastText (for creating output phrases)
        self.ftext = fasttext.FastText()

        # define rnns
        self.num_layers = num_layers_for_rnns
        self.encoder_rnn = EncoderRNN(input_size=vocab_dim,
                                      hidden_size=encoder_hidden_dim,
                                      num_layers=self.num_layers,
                                      ftext = self.ftext)

        self.decoder_rnn = DecoderRNN(hidden_size=decoder_hidden_dim,
                                      output_size=vocab_dim,
                                      num_layers=self.num_layers)

        # define dense modules
        self.encoder_dense = EncoderDense(hidden_dim=encoder_hidden_dim,
                                          z_dim=z_dim)

        self.decoder_dense = DecoderDense(z_dim=z_dim,
                                          hidden_dim=decoder_hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.dataset = dataset
        self.max_length = max_length


    def model(self, input_variable, target_variable, step):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder_dense", self.decoder_dense)
        pyro.module("decoder_rnn", self.decoder_rnn)

        # setup hyperparameters for prior p(z)
        # the type_as ensures we get CUDA Tensors if x is on gpu
        z_mu = ng_zeros([self.num_layers, self.z_dim], type_as=target_variable.data)
        z_sigma = ng_ones([self.num_layers, self.z_dim], type_as=target_variable.data)

        # sample from prior
        # (value will be sampled by guide when computing the ELBO)
        z = pyro.sample("latent", dist.normal, z_mu, z_sigma)

        # init vars
        target_length = target_variable.shape[0]

        decoder_input = dataset.to_onehot([[self.ftext.SOS_index]])
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

        decoder_outputs = np.ones((target_length))
        decoder_hidden = self.decoder_dense(z)

        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder_rnn(
                decoder_input, decoder_hidden)

            if self.use_cuda:
                decoder_outputs[di] = np.argmax(decoder_output.cpu().data.numpy())
            else:
                decoder_outputs[di] = np.argmax(decoder_output.data.numpy())
            
            if TEACHER_FORCING:
                decoder_input = target_variable[di]
            else:
                val = self.dataset.to_onehot(np.array([decoder_outputs[di]]))
                decoder_input = val

            pyro.observe("obs_{}".format(di), dist.bernoulli, target_variable[di], decoder_output[0])

        # ----------------------------------------------------------------
        # prepare offer
        if self.use_cuda:
            offer = np.argmax(input_variable.cpu().data.numpy(), axis=1).astype(int)
        else:
            offer = np.argmax(input_variable.data.numpy(), axis=1).astype(int)

        # prepare answer
        if self.use_cuda:
            answer = np.argmax(target_variable.cpu().data.numpy(), axis=1).astype(int)
        else:
            answer = np.argmax(target_variable.data.numpy(), axis=1).astype(int)

        # prepare rnn
        rnn_response = list(map(int, decoder_outputs))
        
        # print output
        if step % 10 == 0:
            print("---------------------------")
            #print("Offer: ", dataset.to_phrase(offer))
            #print("Answer:", self.dataset.to_phrase(answer))
            #print("RNN:", self.dataset.to_phrase(rnn_response))
            print("Offer: ", ' '.join(self.ftext.get_words_from_indices(offer)))
            print("Answer:", ' '.join(self.ftext.get_words_from_indices(answer)))
            print("RNN:", ' '.join(self.ftext.get_words_from_indices(rnn_response)))
        # ----------------------------------------------------------------


    def guide(self, input_variable, target_variable, step):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder_dense", self.encoder_dense)
        pyro.module("encoder_rnn", self.encoder_rnn)

        # init vars
        context_length = CONTEXT_LENGTH
        input_length = input_variable.shape[0]
        encoder_outputs = Variable(torch.zeros(input_length, self.encoder_rnn.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if USE_CUDA else encoder_outputs
        encoder_hidden = self.encoder_rnn.init_hidden_gru()

        hidden_average = Variable(torch.zeros(self.num_layers, 1, self.encoder_rnn.hidden_size))
        hidden_average = hidden_average.cuda() if USE_CUDA else hidden_average

        # INNER LSTM
        # loop to encode, then average the hidden states
        # to get the final sentence embedding
        for c in range(context_length):
            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder_rnn(
                    input_variable[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0][0]
                hidden_average += encoder_hidden
            hidden_average = hidden_average/input_length

        # OUTER LSTM
        # recurrently encode each of the sentence embeddings

        # use the encoder to get the parameters used to define q(z|x)
        #z_mu, z_sigma = self.encoder_dense(encoder_hidden)
        z_mu, z_sigma = self.encoder_dense(hidden_average)

        # sample the latent code z
        pyro.sample("latent", dist.normal, z_mu, z_sigma)


#
# ===============================================
# ===============================================
# ===============================================
#
# Training loop


num_epochs = 100
test_frequency = 1

vrae = VRAE(dataset,
            VOCAB_SIZE,
            ENCODER_HIDDEN_SIZE,
            Z_DIMENSION,
            DECODER_HIDDEN_SIZE,
            MAX_LENGTH,
            NUM_LAYERS_FOR_RNNS,
            USE_CUDA)
optimizer = optim.Adam({"lr": LEARNING_RATE})
svi = SVI(vrae.model, vrae.guide, optimizer, loss="ELBO")

f=fasttext.FastText()

for epoch in range(30):
    print("Start epoch!")
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x
    # returned by the data loader
    for convo_i in range(dataset.size()):
        x, y = dataset.next_batch()

        #HACK for overfitting
        y = ['SOS', 'this', 'is', 'a', 'test', '.', 'EOS']

        x = f.get_indices(x)
        y = f.get_indices(y)

        x = dataset.to_onehot(x, long_type=False)
        y = dataset.to_onehot(y, long_type=False)
        
        # do ELBO gradient and accumulate loss
        if USE_CUDA:
            loss = svi.step(x.cuda(), y.cuda(), convo_i)
        else:
            loss = svi.step(x, y, convo_i)
        epoch_loss += loss

        # print loss
        if convo_i % 10 == 0:
            print("Epoch: {}, Step: {}, NLL: {}".format(epoch, convo_i, loss))
            print("---------------------------\n")

    print("\n\nTrained epoch: {}, epoch loss: {}\n\n".format(epoch, epoch_loss))
