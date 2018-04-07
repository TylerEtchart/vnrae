import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.autograd import Variable

#from dataset import Dataset
from dataset_fasttext import Dataset
import fasttext
torch.set_printoptions(profile="short")

#
# ===============================================
# ===============================================
# ===============================================
#
# Globals

# init dataset
dataset = Dataset()

# Sizes
#VOCAB_SIZE = dataset.vocab_size
VOCAB_SIZE = 50002
HIDDEN_SIZE = 300 #size of fasttext vectors
#HIDDEN_SIZE = 256
LEARNING_RATE = .001
MAX_LENGTH = 256
USE_CUDA = True

#
# ===============================================
# ===============================================
# ===============================================
#
# RNNs

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        ftext = fasttext.FastText()
        ftextTensor = torch.FloatTensor(ftext.vectors)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding.weight.data.copy_(ftextTensor)
        self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(self.num_layers, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(self.num_layers, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result

#
# ===============================================
# ===============================================
# ===============================================
#
# Training function

def train(input_variable,
          target_variable,
          encoder,
          decoder,
          encoder_optimizer,
          decoder_optimizer,
          criterion,
          epoch,
          convo_i,
          teacher_forcing_ratio=.5,
          max_length=MAX_LENGTH,
          ftext = None):

    encoder_hidden = encoder.init_hidden()
    if USE_CUDA:
        encoder_hidden = encoder_hidden.cuda()
        input_variable = input_variable.cuda()
        target_variable = target_variable.cuda()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = len(input_variable)
    target_length = len(target_variable)

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if USE_CUDA else encoder_outputs

    decoder_outputs = -np.ones((max_length))

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[dataset.SOS_index]]))
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_outputs[di] = ni
            # out_char = decoder_output.data.numpy()[0]
            # out_char = out_char / np.sum(out_char)
            # decoder_outputs[di] = np.random.choice(range(VOCAB_SIZE), p=out_char)

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

            loss += criterion(decoder_output, target_variable[di])

            decoder_outputs[di] = ni
            # out_char = decoder_output.data.numpy()[0]
            # out_char = out_char / np.sum(out_char)
            # decoder_outputs[di] = np.random.choice(range(VOCAB_SIZE), p=out_char)

            if ni == ftext.EOS_index:
                break


    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    input_variable = input_variable.view(-1)
    if USE_CUDA:
        # prepare offer
        offer = input_variable.cpu().data.numpy()
        
        # prepare answer
        target_variable = target_variable.view(-1)
        answer = target_variable.cpu().data.numpy()
    else:
        # prepare offer
        offer = input_variable.data.numpy()

        # prepare answer
        target_variable = target_variable.view(-1)
        answer = target_variable.data.numpy()

    # prepare rnn
    for i in range(len(decoder_outputs)):
        if decoder_outputs[i] == -1:
            break
    formated_decoder_outputs = decoder_outputs[:i]
    rnn_response = list(map(int, formated_decoder_outputs))

    if convo_i%100 == 0:
        print("\n---------------------------")
        print("Epoch: {}, Step: {}".format(epoch, convo_i))
        print("Loss:   {}".format(loss.data[0] / target_length))
        print("---------------------------\n")
        if ftext is not None:
            print("Offer: ", ' '.join(ftext.get_words_from_indices(offer)))
            print("Answer:", ' '.join(ftext.get_words_from_indices(answer)))
            print("RNN:", ' '.join(ftext.get_words_from_indices(rnn_response)))
        sys.stdout.flush()


#
# ===============================================
# ===============================================
# ===============================================
#
# Training loop


# define rnns
encoder_rnn = EncoderRNN(input_size=VOCAB_SIZE,
                         hidden_size=HIDDEN_SIZE,
                         num_layers=1)


decoder_rnn = DecoderRNN(hidden_size=HIDDEN_SIZE,
                         output_size=VOCAB_SIZE,
                         num_layers=1)

if USE_CUDA:
    encoder_rnn=encoder_rnn.cuda()
    decoder_rnn=decoder_rnn.cuda()

# loss
criterion = nn.NLLLoss()

encoder_params = filter(lambda p: p.requires_grad, encoder_rnn.parameters())
encoder_optimizer = optim.SGD(encoder_params, lr=LEARNING_RATE)
#encoder_optimizer = optim.SGD(encoder_rnn.parameters(), lr=LEARNING_RATE)
decoder_optimizer = optim.SGD(decoder_rnn.parameters(), lr=LEARNING_RATE)

# main conversation loop
# x, y = dataset.next_batch()

f = fasttext.FastText()
for epoch in range(30):
#for epoch in range(10):
    print("Running for " + str(dataset.size()) + " steps...")
    for convo_i in range(dataset.size()):
    #for convo_i in range(100):
        # translation loop vars
        x, y = dataset.next_batch()
        
        #HACK to check overfit
        #y = ['SOS','this','is','a','test','.','EOS']

        x = f.get_indices(x)
        y = f.get_indices(y)

        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        
        x2 = Variable(torch.LongTensor(x))
        y2 = Variable(torch.LongTensor(y))

        train(Variable(torch.LongTensor(x)),
            Variable(torch.LongTensor(y)),
            encoder_rnn,
            decoder_rnn,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
            epoch,
            convo_i,
            ftext = f)
