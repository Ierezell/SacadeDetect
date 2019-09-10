import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# #####################
# Training parameters #
# #####################
NB_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
HIDDEN_SIZE = 256


ROOT_WEIGHTS = './weights/'
ROOT_IMAGE = './images/'
ROOT_DATASET = './Dataset/events.json'

ATTN_MODEL = 'fullyconnected'  # 'fullyconnected' 'concat'
N_LAYERS = 1
DROPOUT = 0.2

PRINT_EVERY = 1
NUM_WORKERS = 4
TIME_LIMIT = 10000
NB_EVENT_LIMIT = 20
MIN_EVENT_SIZE = 200

CONFIG = {"N_LAYERS": N_LAYERS,
          "DROPOUT": DROPOUT,
          "NB_EVENT_LIMIT": NB_EVENT_LIMIT,
          "MIN_EVENT_SIZE": MIN_EVENT_SIZE,
          "ATTN_MODEL": ATTN_MODEL,
          "HIDDEN_SIZE": HIDDEN_SIZE,
          "LEARNING_RATE": LEARNING_RATE,
          "BATCH_SIZE": BATCH_SIZE,
          "NB_EPOCHS": NB_EPOCHS,
          "TIME_LIMIT": TIME_LIMIT/100,
          }

folder_weights = str(CONFIG["N_LAYERS"]) + "_" + \
    str(CONFIG["DROPOUT"])+"_" +\
    str(CONFIG["NB_EVENT_LIMIT"])+"_" +\
    str(CONFIG["MIN_EVENT_SIZE"])+"_" +\
    str(CONFIG["ATTN_MODEL"])+"_" +\
    str(CONFIG["HIDDEN_SIZE"])+"_" +\
    str(CONFIG["LEARNING_RATE"])+"_" +\
    str(CONFIG["BATCH_SIZE"])+"_" +\
    str(CONFIG["TIME_LIMIT"])+"_" +\
    str(CONFIG["NB_EPOCHS"])+'/'

if not os.path.exists(ROOT_WEIGHTS + folder_weights):
    os.makedirs(ROOT_WEIGHTS + folder_weights)
    LOAD_PREVIOUS = False
else:
    LOAD_PREVIOUS = False

PATH_WEIGHTS_RNN = ROOT_WEIGHTS+folder_weights+'Rnn.pt'
PATH_WEIGHTS_CLASSIFIER = ROOT_WEIGHTS+folder_weights+'Classifier.pt'
PATH_WEIGHTS_AUTOENCODER = ROOT_WEIGHTS + folder_weights + 'Autoencoder.pt'
