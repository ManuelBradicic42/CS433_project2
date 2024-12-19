import numpy as np
import csv
'''
Class responsible for tokenising words into numberical values
e.g. Rab8 --> [18 19 20 11]
'''
class Tokenizer:
    def __init__(self, vocab, max_size):
        self.vocab = vocab
        self.vocab['_'] = 0
        self.rev_vocab = {v: k for k, v in vocab.items()}
        self.max_size = max_size
    
    def encode(self, text):
        while len(text) < self.max_size:
            text += "_"
        return np.array([self.vocab[char] for char in text])
    
    def decode(self, tokens):
        return ''.join([self.rev_vocab[token] for token in tokens])


def get_bag_of_characters(path, return_max=False):
    bag_of_strings = np.array([])
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            bag_of_strings = np.append(bag_of_strings, row[0])

    unique_characters = set(''.join(bag_of_strings))
    unique_string = ''.join(sorted(unique_characters))

    if return_max:
        max_size = max([len(x) for x in bag_of_strings])
        return unique_string, max_size
    else:
        return unique_string