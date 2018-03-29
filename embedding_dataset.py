from collections import Counter
import numpy as np
from scipy import spatial
import codecs
import torch
from torch.autograd import Variable
import gensim
from timeit import default_timer as timer
import _pickle as pickle


class EmbeddingDataset:


    def __init__(self, load=False, filename="ijcnlp_dailydialog/dialogues_text.txt"):
        if load:
            self.load_vocab()
        else:
            self.generate_vocab(filename)
        self.generate_conversations(filename)
        self.create_batches()
        # self.reset_batch_pointer()


    def generate_vocab(self, filename):
        print("Loading model...")
        start = timer()
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)  
        end = timer()
        print("Load complete! It took {}s".format(end - start))

        with open(filename, "r") as f:
            data = f.read()
        words = data.split()
        filtered_words = []
        for w in words:
            try:
                temp = w2v_model[w]
                filtered_words.append(w)
            except:
                pass

        counter = Counter(filtered_words)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])

        self.words, _ = zip(*count_pairs)
        self.words = list(self.words)
        self.vocab_size = len(self.words)

        # generate vocab via gensim
        self.vocab = {}
        for w in self.words:
            self.vocab[w] = w2v_model[w]

        with open("data/words.pkl", "wb") as f:
            pickle.dump(self.words, f)
        with open("data/vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f)


    def load_vocab(self):
        with open("data/words.pkl", "rb") as f:
            self.words = pickle.load(f)
            self.vocab_size = len(self.words)
        with open("data/vocab.pkl", "rb") as f:
            self.vocab = pickle.load(f)

    
    def generate_conversations(self, filename):
        with open(filename, "r") as file:
            text_content = file.readlines()
            
        self.conversations = []
        for line_no in range(len(text_content)):
            text = text_content[line_no].split("__eou__")
            text = [t.strip() for t in text if t.strip() != ""]

            for turn in range(len(text) - 1):
                offer = text[turn].split()
                offer = [o for o in offer if o in self.words]
                answer = text[turn+1].split()
                answer = [a for a in answer if a in self.words]
                self.conversations.append((offer, answer))


    def create_batches(self):
        self.encoder_turns = []
        self.decoder_turns = []
        self.encoded_conversations = []
        longest_possible_sequence = 200
        for convo in self.conversations:
            if len(convo[0]) > longest_possible_sequence or \
                    len(convo[1]) > longest_possible_sequence:
                continue

            print(encoder_turn)
            input("stop")

            encoder_turn = list(map(self.vocab.get, convo[0]))
            encoder_turn.insert(0, self.SOS_index)
            encoder_turn.append(self.EOS_index)
            encoder_turn = np.array(encoder_turn)

            decoder_turn = list(map(self.vocab.get, convo[1]))
            decoder_turn.insert(0, self.SOS_index)
            decoder_turn.append(self.EOS_index)
            decoder_turn = np.array(decoder_turn)

            self.encoded_conversations.append((encoder_turn, decoder_turn))
            self.encoder_turns.append(encoder_turn)
            self.decoder_turns.append(decoder_turn)


    def next_batch(self):
        encoder, decoder = self.encoder_turns[self.pointer], self.decoder_turns[self.pointer]
        self.pointer += 1
        if self.pointer >= len(self.encoder_turns):
            self.reset_batch_pointer()
        return encoder, decoder


    def reset_batch_pointer(self):
        self.pointer = 0


    def get_word(self, vec):
        words = []
        for w in self.vocab.keys():
            distance = spatial.distance.cosine(vec, self.vocab[w])
            words.append((w, distance))
        ranked = sorted(words, key=lambda x: x[1])
        return ranked[0][0]


    def get_vec(self, word):
        return self.vocab[word]


    def to_onehot(self, x, long_type=False):
        onehot_stack = torch.zeros((len(x), self.vocab_size))
        onehot_stack[np.array(range(len(x))), x] = 1
        if long_type:
            onehot_stack = onehot_stack.type(torch.LongTensor)
        return Variable(onehot_stack)


    def to_phrase(self, x):
        return "".join([self.chars[x[i]] for i in range(len(x))])


    def size(self):
        return len(self.encoder_turns)



if __name__ == "__main__":
    dataset = EmbeddingDataset(load=True)
    print("Done!")
    v = dataset.get_vec("equalization")
    print(dataset.get_word(v))
    # x, y = dataset.next_batch()

    # # print the indicies
    # print(x)
    # print(y)

    # # print the chars
    # print("".join([dataset.chars[x[i]] for i in range(len(x))]))
    # print("".join([dataset.chars[y[i]] for i in range(len(y))]))


    emb = False
    if emb:
        print("\n------------------------------")
        print("Loading model...")
        start = timer()
        model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)  
        end = timer()
        print("Load complete! It took {}s".format(end - start))

        # model = gensim.models.Word2Vec()
        # model.build_vocab_from_freq({"Word1": 15, "Word2": 20})
        print("Printing data...")
        print(model)
        print(model['pie'])
        print(len(model['pie']))