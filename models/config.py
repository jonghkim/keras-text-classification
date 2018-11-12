class ConvConfig():
    
    MODE = "train"

    BATCH_SIZE = 512  # Batch size for training.
    EPOCHS = 100  # Number of epochs to train for.
    LATENT_DIM = 256  # Latent dimensionality of the encoding space.
    NUM_SAMPLES = 700  # Number of samples to train on. 
    MAX_SEQUENCE_LENGTH = 200
    MAX_NUM_WORDS = 20000
    EMBEDDING_DIM = 100
    CLASS_NUM = 2

    PREDICT_SAMPLE_SIZE = 5
    PREDICTION_BATCH_SIZE = 32

    DROPOUT = 0.2

    INPUT_MAX_LEN = 60

    def set_params(self, params):
        for key, item in params.items():
            if key == 'MODE':
                self.MODE = item
            elif key == 'BATCH_SIZE':
                self.BATCH_SIZE = item
            elif key == 'EPOCHS':                
                self.EPOCHS = item
            elif key == 'LATENT_DIM':                
                self.LATENT_DIM = item
            elif key == 'LATENT_DIM_DECODER':                
                self.LATENT_DIM_DECODER = item
            elif key == 'NUM_SAMPLES':                
                self.NUM_SAMPLES = item
            elif key == 'MAX_SEQUENCE_LENGTH':                
                self.MAX_SEQUENCE_LENGTH = item
            elif key == 'MAX_NUM_WORDS':                
                self.MAX_NUM_WORDS = item
            elif key == 'EMBEDDING_DIM':                
                self.EMBEDDING_DIM = item   
            elif key == 'ADVERSARIAL':
                self.ADVERSARIAL = item
            elif key == 'STYLE_TRANSFER':
                self.STYLE_TRANSFER = item
            elif key == 'CLASS_NUM':
                self.CLASS_NUM = item 
            elif key == 'CLASS_DIM':                
                self.CLASS_DIM = item
            elif key == 'PREDICTION_BATCH_SIZE':            
                self.PREDICTION_BATCH_SIZE = item
            elif key == 'SUCCESS_CRITERIA':
                self.SUCCESS_CRITERIA = item
            elif key == 'BINARY_PREDICTION':
                self.BINARY_PREDICTION = item            
            else:
                print("unknown param", key)
                
    def __repr__(self):
        arttribute = vars(self)
        arttribute = {k:v for k,v in arttribute.items() if not k.startswith("__")}
        return str(arttribute)