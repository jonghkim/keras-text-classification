import os, sys
import numpy as np

from utils.data_helper import DataHelper
from models.config import ConvConfig
from models.conv_model import ConvModel

if __name__ == "__main__":
    config = ConvConfig()
    params = {'MODE':'eval'}    
    config.set_params(params)
    
    NEG_DATA_PATH = 'data/sentiment/neg.txt' 
    POS_DATA_PATH = 'data/sentiment/pos.txt'
    WORD2VEC_PATH = '/data/pretrained_model/word_embedding/glove.6B/glove.6B.%sd.txt' % config.EMBEDDING_DIM
    LOAD_PATH = 'models/models/model_w{}_e{}_c{}.h5'.format(config.EMBEDDING_DIM, config.LATENT_DIM, config.CLASS_NUM)

    print(config)
    print("Data Path: ", NEG_DATA_PATH, POS_DATA_PATH)
    print("Word2Vec Path: ", WORD2VEC_PATH)
    print("Save Path: ", WORD2VEC_PATH)

    data_helper = DataHelper(config)

    #### load the data #### 
    input_texts, target_texts, target_texts_inputs, classes = data_helper.read_txt_sentiment(NEG_DATA_PATH, POS_DATA_PATH)

    #### tokenize the inputs, outputs ####
    input_sequences, word2idx_inputs, max_len_input = \
                         data_helper.create_vocab(input_texts, target_texts, target_texts_inputs)
                         

    #### set data of model ####
    model = ConvModel(config)
    model.set_data(input_sequences, classes, max_len_input, word2idx_inputs)

    #### build model ####
    model.predict_build_model(LOAD_PATH)    
    model.predict_sample(input_texts, classes)

    ans = input("Save Predictions? [Y/n]")
    if ans and ans.lower().startswith('n'):
        sys.exit("**** Evaluation Done ****")