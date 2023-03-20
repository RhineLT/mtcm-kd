import torch

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