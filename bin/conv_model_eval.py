import os, sys
import numpy as np

from utils import parameter_setup
from utils.data_helper import DataHelper
from models.config import ConvConfig
from models.conv_model import ConvModel

if __name__ == "__main__":
    config = ConvConfig()
    params = {'MODE':'eval'}    
    config.set_params(params)
