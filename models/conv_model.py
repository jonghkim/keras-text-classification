import os, sys
import numpy as np
import pandas as pd

import keras
from keras.models import Model
from keras.layers import (Input, LSTM, GRU, Dense, Embedding, Bidirectional,
                            Conv1D, MaxPooling1D, Merge, Dropout, Flatten,
                            Concatenate, BatchNormalization)
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

        self.steps_per_epoch = int(len(self.input_sequences)/self.config.BATCH_SIZE)

        print('---- Set Data Finished ----')

    def batch_get_input(self, idx, batch_size):
        batch_input_sequences = self.input_sequences[idx:idx+batch_size]
        batch_x = [batch_input_sequences]

        return batch_x
                
    def batch_get_output(self, idx, batch_size):
        # create targets, since we cannot use sparse
        # categorical cross entropy when we have sequences
        classes_targets_one_hot = np.zeros((
                                                    batch_size,
                                                    self.config.CLASS_NUM
                                                ),
                                                dtype='float32'
                                            )
        # assign the values
        for i, d in enumerate(self.classes[idx:idx+batch_size]):
            classes_targets_one_hot[i, d] = 1

        y = classes_targets_one_hot

        return y

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
        # create embedding layer
        embedding_layer = Embedding(self.embedding_matrix.shape[0],
                                    self.config.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.max_len_input,
                                    name='input_embedding'
                                    # trainable=True
                                    )
        ##### build the model #####
        inputs_placeholder = Input(shape=(self.max_len_input,),name='input')
        embedded_sequence = embedding_layer(inputs_placeholder)

        convs = []
        filter_sizes = [3,4,5]
        for filter_size in filter_sizes:
            conv_layer = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequence)
            bn_layer = BatchNormalization()(conv_layer)
            pooling_layer = MaxPooling1D(pool_size=3)(bn_layer)
            convs.append(pooling_layer)
        merge = Merge(mode='concat', concat_axis=1)(convs)
        z = Flatten()(merge)
        dropout = Dropout(self.config.DROPOUT)(z)
        output = Dense(128, activation='relu')(dropout)
        output = Dropout(self.config.DROPOUT)(output)
        output = Dense(self.config.CLASS_NUM, activation='softmax')(dropout)

        model = Model(inputs=inputs_placeholder, outputs=output)

        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['acc'])

        model.summary()
        print('---- Build Model Finished ----')

        self.model = model

    def train_model(self):
        #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
        #callbacks_list = [early_stopping]
        callbacks_list = []

        r = self.model.fit_generator(generator=self.batch_generator(),
                    epochs=self.config.EPOCHS,
                    steps_per_epoch=self.steps_per_epoch,
                    verbose=1,                    
                    use_multiprocessing=False,
                    workers=1,
                    callbacks=callbacks_list)
        print('---- Train Model Finished ----')        

    def save_model(self, SAVE_PATH):
        self.model.save(SAVE_PATH)
        print('---- Save Model Finished ----')        

    def predict_build_model(self, LOAD_PATH):
        self.model = load_model(LOAD_PATH)

        print('---- Load Model Finished ----')        

    def predict_sequences(self, batch_input_sequences):

        predicted_classes = self.model.predict_classes(batch_input_sequences, batch_size=self.config.PREDICTION_BATCH_SIZE, verbose=1)

        return predicted_classes

    def predict_sample(self, input_texts, classes):
        # map indexes back into real words
        # so we can view the results
        while True:
            # Do some test translations
            i = np.random.choice(len(input_texts)-self.config.PREDICT_SAMPLE_SIZE+1)
            
            batch_size = len(self.input_sequences[i:i+self.config.PREDICT_SAMPLE_SIZE])
            batch_input_sequences = self.batch_get_input(i, batch_size)

            predicted_classes = self.predict_sequences(batch_input_sequences)

            for j in range(batch_size):
                print('-')
                print('*** Input:', input_texts[i+j], ' ***')
                print('   Predicted Class:', ' '.join(predicted_classes[j]))
                print('Actual Class:', classes[i+j])

            ans = input("Continue? [Y/n]")
            if ans and ans.lower().startswith('n'):
                break