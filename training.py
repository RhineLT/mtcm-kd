import torch
from torchvision import transforms as t
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.tensorboard import SummaryWriter


from metrics import calculate_dice_score, calculate_hd95_multi_class, save_history
from models import ResUNET_channel_attention
from loss_functions import dice_loss, jaccard_loss, CrossEntropyLoss, KL_divergence
from optimizer import Ranger


## config dictionary for model training
config = {
    "model_name": "ResUNET_channel_attention",
    "model_path": "./saved_models/",
    "data_path": "./BraTS_Dataset/",
    "batch_size": 4,
    "num_epochs": 100,
    "image_height": 128,
    "image_width": 128,
    "image_depth": 128,
    "model_params": {
        "in_channels": 1,
        "out_channels": 4,
        "dropout": 0.3,
        "attention": True,
        "attention_type": "channel",
        "learning_rate": 0.0001,
        },
}

## device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


