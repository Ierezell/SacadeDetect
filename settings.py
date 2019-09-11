import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNN = "LSTM"

# #####################
# Training parameters #
# #####################
NB_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
HIDDEN_SIZE = 512


ROOT_WEIGHTS = './weights/'
ROOT_IMAGE = './images/'
ROOT_DATASET = './Dataset/events.json'

ATTN_MODEL = 'fullyconnected'  # 'fullyconnected' 'concat'
N_LAYERS = 2
DROPOUT = 0.2

NUM_WORKERS = 4
TIME_LIMIT = 100000
NB_EVENT_LIMIT = 20
MIN_EVENT_SIZE = 200

CONFIG = {"N_LAYERS": str(N_LAYERS),
          "DROPOUT": str(DROPOUT),
          "NB_EVENT_LIMIT": str(NB_EVENT_LIMIT),
          "MIN_EVENT_SIZE": str(MIN_EVENT_SIZE),
          "ATTN_MODEL": str(ATTN_MODEL),
          "HIDDEN_SIZE": str(HIDDEN_SIZE),
          "LEARNING_RATE": str(LEARNING_RATE),
          "BATCH_SIZE": str(BATCH_SIZE),
          "NB_EPOCHS": str(NB_EPOCHS),
          "TIME_LIMIT": str(TIME_LIMIT / 1000),
          "RNN": str(RNN),
          }

folder_weights = CONFIG["N_LAYERS"] + "_" + CONFIG["DROPOUT"]+"_" +\
    CONFIG["NB_EVENT_LIMIT"]+"_" + CONFIG["MIN_EVENT_SIZE"]+"_" +\
    CONFIG["ATTN_MODEL"]+"_" + CONFIG["HIDDEN_SIZE"]+"_" +\
    CONFIG["LEARNING_RATE"]+"_" + CONFIG["BATCH_SIZE"]+"_" +\
    CONFIG["TIME_LIMIT"]+"_" + CONFIG["NB_EPOCHS"]+CONFIG["RNN"]+'/'

if not os.path.exists(ROOT_WEIGHTS + folder_weights):
    os.makedirs(ROOT_WEIGHTS + folder_weights)
    LOAD_PREVIOUS = False
else:
    LOAD_PREVIOUS = True

PATH_WEIGHTS_RNN = ROOT_WEIGHTS+folder_weights+'Rnn.pt'
PATH_WEIGHTS_CLASSIFIER = ROOT_WEIGHTS+folder_weights+'Classifier.pt'
PATH_WEIGHTS_AUTOENCODER = ROOT_WEIGHTS + folder_weights + 'Autoencoder.pt'
