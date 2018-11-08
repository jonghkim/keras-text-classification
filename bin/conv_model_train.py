import os, sys
import numpy as np

from utils.data_helper import DataHelper
from models.config import ConvConfig
from models.conv_model import ConvModel

if __name__ == "__main__":
    config = ConvConfig()
    NEG_DATA_PATH = 'data/sentiment/neg.txt' 
    POS_DATA_PATH = 'data/sentiment/pos.txt'
    WORD2VEC_PATH = '/data/pretrained_model/word_embedding/glove.6B/glove.6B.%sd.txt' % config.EMBEDDING_DIM
    SAVE_PATH = 'models/models/model_w{}_e{}_c{}.h5'.format(config.EMBEDDING_DIM, config.LATENT_DIM, config.CLASS_DIM)

    print(config)
    print("Data Path: ", NEG_DATA_PATH, POS_DATA_PATH)
    print("Word2Vec Path: ", WORD2VEC_PATH)
    print("Save Path: ", SAVE_PATH)

    data_helper = DataHelper(config)

    #### load the data #### 
    input_texts, target_texts, target_texts_inputs, classes = data_helper.read_txt_sentiment(NEG_DATA_PATH, POS_DATA_PATH)

    #### tokenize the inputs, outputs ####
    input_sequences, word2idx_inputs, max_len_input = \
                         data_helper.create_vocab(input_texts, target_texts, target_texts_inputs)
                         
    #### load word2vec pretrained model ####
    word2vec = data_helper.load_word2vec(WORD2VEC_PATH)

    #### create embedding matrix ####
    embedding_matrix = data_helper.create_embedding_matrix(word2vec, word2idx_inputs, WORD2VEC_PATH)

    #### set data of model ####
    model = ConvModel(config)
    model.set_data(input_sequences, classes, max_len_input, word2idx_inputs)
    model.set_embedding_matrix(embedding_matrix)
    #### build model ####
    model.build_model()    
    #### train model ####
    model.train_model()
    #### save model ####
    model.save_model(SAVE_PATH)