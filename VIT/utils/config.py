import torch
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(0)
np.random.seed(0)
random.seed(42)

### Constants
SCRATCH = False
BATCH_SIZE = 64         # Training batch size
LR = 1e-4               # Learning rate of model
NUM_EPOCHES = 10        # Number of Epochs
LOSS_NAME = "CTC"       # Loss of interest: ["CTC", "cross_entropy"]

### Paths 
path_chess_moves = "../src/data_generation/all_moves_proba.txt"
path_images = "../data/chess_images"
path_csv = "../data/test_data/val.csv"
path_test_images = "../data/test_data/images"

### Additions

if SCRATCH:
    IMG_DIM = (96, 96)    # Image dimensions
else:
    IMG_DIM = (224, 224)    # Image dimensions


loss_hist = {
    "train_loss" : [],
    "val_loss" : [],
    "train_loss_iterations" : [],
    "test_acc": [],
    "test_cer": [],
    "config" : {
        'loss' : LOSS_NAME,
        'bs' : BATCH_SIZE,
        'lr' : LR,
        'epochs' : NUM_EPOCHES,
        'imgdim' : IMG_DIM   
        },
}