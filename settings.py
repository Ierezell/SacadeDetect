import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# #####################
# Training parameters #
# #####################
NB_EPOCHS = 1000
BATCH_SIZE = 1
LEARNING_RATE = 5e-3
HIDDEN_SIZE = 2048

ATTN_MODEL = 'fullyconnected'  # 'fullyconnected' 'concat'
N_LAYERS = 4
DROPOUT = 0.1

LOAD_PREVIOUS = True
PRINT_EVERY = 1
NUM_WORKERS = 12

# #############
# Directories #
# #############
PATH_DATASET = './Dataset/events.json'
ROOT_WEIGHTS = './weights/'
