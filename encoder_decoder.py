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
# Globals

# init dataset
dataset = Dataset()

# Sizes
VOCAB_SIZE = dataset.vocab_size
HIDDEN_SIZE = 256
LEARNING_RATE = .001
MAX_LENGTH = 256
USE_CUDA = False

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

        self.embedding = nn.Embedding(input_size, hidden_size)
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
          max_length=MAX_LENGTH):

    encoder_hidden = encoder.init_hidden()

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

            if ni == dataset.EOS_index:
                break


    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    # prepare offer
    input_variable = input_variable.view(-1)
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

    print("\n---------------------------")
    print("Epoch: {}, Step: {}".format(epoch, convo_i))
    print("Loss:   {}".format(loss.data[0] / target_length))
    print("---------------------------\n")
    print("Offer: ", dataset.to_phrase(offer))
    print("Answer:", dataset.to_phrase(answer))
    print("RNN:", dataset.to_phrase(rnn_response))


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

# loss
criterion = nn.NLLLoss()
encoder_optimizer = optim.SGD(encoder_rnn.parameters(), lr=LEARNING_RATE)
decoder_optimizer = optim.SGD(decoder_rnn.parameters(), lr=LEARNING_RATE)

# main conversation loop
# x, y = dataset.next_batch()
for epoch in range(30):
    for convo_i in range(dataset.size()):
        # translation loop vars
        x, y = dataset.next_batch()
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
        train(Variable(torch.LongTensor(x)),
            Variable(torch.LongTensor(y)),
            encoder_rnn,
            decoder_rnn,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
            epoch,
            convo_i)
