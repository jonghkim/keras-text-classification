from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import os, sys
import pandas as pd
import gensim

class DataHelper():
    def __init__(self, config):
        self.config = config

    def read_txt_sentiment(self, NEG_DATA_PATH, POS_DATA_PATH):
        # Where we will store the data
        neg_texts = [] # sentence in original language
        pos_texts = [] # sentence in target language
        classes = []

        # load in the data
        # download the data at: http://www.manythings.org/anki/
        t = 0
        for line in open(NEG_DATA_PATH):
            # only keep a limited number of samples
            t += 1
            if t > self.config.NUM_SAMPLES:
                break
            # split up the input and translation
            neg_text = line.rstrip()
            neg_texts.append(neg_text)
            classes.append(0)
                
        t = 0    
        for line in open(POS_DATA_PATH):
            # only keep a limited number of samples
            t += 1
            if t > self.config.NUM_SAMPLES:
                break
            # split up the input and translation
            pos_text = line.rstrip()
            pos_texts.append(pos_text)
            classes.append(1)
            
        print("num samples:", len(neg_texts)+len(pos_texts))

        input_texts = neg_texts + pos_texts

        target_texts = []
        target_texts_inputs = []

        for input_text in input_texts:
            target_texts.append(input_text+' <eos>')
            target_texts_inputs.append('<sos> ' + input_text)    
        
        return input_texts, target_texts, target_texts_inputs, classes

    def create_vocab(self, input_texts, target_texts, target_texts_inputs):
        # tokenize the inputs
        tokenizer_inputs = Tokenizer(num_words=self.config.MAX_NUM_WORDS)
        tokenizer_inputs.fit_on_texts(input_texts)
        input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

        # get the word to index mapping for input language
        word2idx_inputs = tokenizer_inputs.word_index
        print('Found %s unique input tokens.' % len(word2idx_inputs))

        # determine maximum length input sequence
        # max_len_input = max(len(s) for s in input_sequences)
        max_len_input = self.config.INPUT_MAX_LEN

        # pad the sequences
        input_sequences = pad_sequences(input_sequences, maxlen=max_len_input)
        print("input_sequences.shape:", input_sequences.shape)
        print("input_sequences[0]:", input_sequences[0])

        return (input_sequences, word2idx_inputs, max_len_input)

    def load_word2vec(self, WORD2VEC_PATH):
        # store all the pre-trained word vectors
        print('Loading word vectors...')

        if 'glove' in WORD2VEC_PATH:
            word2vec = {}
            with open(WORD2VEC_PATH) as f:
                # is just a space-separated text file in the format:
                # word vec[0] vec[1] vec[2] ...
                for line in f:
                    values = line.split()
                    word = values[0]
                    vec = np.asarray(values[1:], dtype='float32')
                    word2vec[word] = vec
            print('Found %s word vectors.' % len(word2vec))
        else:
            word2vec = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=False)    

        return word2vec

    def create_embedding_matrix(self, word2vec, word2idx_inputs, WORD2VEC_PATH):
        # prepare embedding matrix
        print('Filling pre-trained embeddings...')
        num_words = min(self.config.MAX_NUM_WORDS, len(word2idx_inputs) + 1)

        embedding_matrix = np.zeros((num_words, self.config.EMBEDDING_DIM))
        for word, i in word2idx_inputs.items():
            if i < self.config.MAX_NUM_WORDS:
                if 'glove' in WORD2VEC_PATH:
                    embedding_vector = word2vec.get(word)
                else: 
                    embedding_vector = word2vec[word] if word in word2vec else None

                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                else:
                    embedding_matrix[i] = np.random.rand(self.config.EMBEDDING_DIM)

        return embedding_matrix