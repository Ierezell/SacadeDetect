import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# #####################
# Training parameters #
# #####################
NB_EPOCHS = 20
BATCH_SIZE = 1
LEARNING_RATE = 5e-5
HIDDEN_SIZE = 512


ROOT_WEIGHTS = './weights/'
ROOT_IMAGE = './images/'
ROOT_DATASET = './Dataset/events.json'

PATH_WEIGHTS_RNN = ROOT_WEIGHTS+'Rnn.pt'
PATH_WEIGHTS_CLASSIFIER = ROOT_WEIGHTS+'Classifier.pt'
PATH_WEIGHTS_AUTOENCODER = ROOT_WEIGHTS + 'Autoencoder.pt'

LOAD_PREVIOUS = False

ATTN_MODEL = 'fullyconnected'  # 'fullyconnected' 'concat'
N_LAYERS = 1
DROPOUT = 0.2

PRINT_EVERY = 1
NUM_WORKERS = 0
TIME_LIMIT = 10000
NB_EVENT_LIMIT = 20
