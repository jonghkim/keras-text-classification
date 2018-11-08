import os, sys
import numpy as np
import pandas as pd

import keras
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, Bidirectional, Conv1D, MaxPooling1D, Merge, Dropout, Flatten, 
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping
import keras.backend as K
if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU

import gensim
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import re
import codecs
import matplotlib.pyplot as plt
from subprocess import check_output
import pickle

from models.config import ConvConfig

class ConvModel():

    def __init__(self, config):
        assert config.MODE in ["train", "eval", "inference"]
        self.train_phase = config.MODE == "train"
        self.config = config

    def set_data(self, input_sequences, classes, max_len_input, word2idx_inputs):

        self.input_sequences =input_sequences

        self.max_len_input = max_len_input
        self.word2idx_inputs = word2idx_inputs

        self.idx2word_eng = {v:k for k, v in self.word2idx_inputs.items()}
        
        self.classes = classes

        print('---- Set Data Finished ----')

    def batch_get_input(self, idx, batch_size):
        batch_input_sequences = self.input_sequences[idx:idx+batch_size]
        batch_x = [batch_input_sequences]

        return batch_x
                
    def batch_get_output(self, idx, batch_size):
        # create targets, since we cannot use sparse
        # categorical cross entropy when we have sequences
        batch_classes_targets_one_hot = np.zeros((
                                                    batch_size,
                                                    self.config.CLASS_NUM
                                                ),
                                                dtype='float32'
                                            )
        # assign the values
        for i, d in enumerate(self.classes[idx:idx+batch_size]):
            batch_classes_targets_one_hot[i, d] = 1

        batch_y = batch_classes_targets_one_hot

        return batch_y

    def batch_generator(self):
        idx = 0
        max_idx = len(self.input_sequences)

        while True:
            if (idx+self.config.BATCH_SIZE) > max_idx:
                idx =0

            batch_size = len(self.input_sequences[idx:idx+self.config.BATCH_SIZE])
            batch_x = self.batch_get_input(idx, batch_size)
            batch_y = self.batch_get_output(idx, batch_size)
            idx = idx + batch_size
            yield(batch_x, batch_y)

    def set_embedding_matrix(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix
        print('---- Set Embedding Matrix Finished ----')

    def build_model(self):
        pass
        
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['acc'])
        model.summary()
        return model

    def train_model(self):
        pass

    def save_model(self):
        pass

    def predict_build_model(self):
        pass

    def predict(self, input_texts, target_texts):
        pass