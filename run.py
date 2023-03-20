
import os

import torch
from torchvision import transforms as t
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.tensorboard import SummaryWriter



from training import  Fit
from models import ResUNET_channel_attention
from loss_functions import dice_loss, jaccard_loss, CrossEntropyLoss, KL_divergence
from optimizer import Ranger
from dataset import  get_loaders, spliting_data_5_folds, reshape_for_deep_supervision, reshape_3d




def run(config):

    ## device configuration
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## settings
    EPOCHS = config["num_epochs"]
    HEIGHT = config["image_height"]
    WEIGHT = config["image_width"]
    DEPTH = config["image_depth"]

    BATCH_SIZE = config["batch_size"]
    LEARNING_RATE = config["model_params"]["learning_rate"]



    ## data loading and processing
    reshape = reshape_3d(HEIGHT, WEIGHT, DEPTH)
    def reshape_volume(x): return reshape(x)
    
    ## transforms
    general_transforms = t.Compose([
       t.Lambda(reshape_volume),
    ])

    ## spliting data into 5 folds holdout validation
    ## check for the folds_data.json file, if not exist, create it
    if not os.path.exists(config["data_split_path"]):
        os.makedirs(config["data_split_path"])
        data_split = spliting_data_5_folds(dataset_dir=config["data_path"])
        
        ## convert the dictionary to json file
        dict_to_json = json.dumps(data_split)
        
        ## save the json file
        with open(config["data_split_path"] + "folds_data.json", "w") as file:
            file.write(dict_to_json)
    else:
        with open(config["data_split_path"] + "folds_data.json", "r") as file:
            data_split = json.load(file)
            
            
          
   
    

    dice_loss_fn = dice_loss
    jaccard_loss_fn = jaccard_loss
    CrossEntropyLoss_fn = CrossEntropyLoss
    
    for fold_index in range(5):
        
         ## get the data loaders
        train_dl, validation_dl, test_dl = get_loaders(
            dataset_dir=config["data_path"],
            batch_size=BATCH_SIZE,
            data_dict=data_split[fold_index],
            train_images_transform = general_transforms,
            train_masks_transform = general_transforms,
            valid_images_transform = general_transforms,
            valid_masks_transform = general_transforms,
            test_images_transform = general_transforms,
            test_masks_transform = general_transforms,
            )
        
        

        ## model configuration
        writer = SummaryWriter(log_dir=config["writer_path"] + config["model_name"] + f"\\fold_{fold_index}")
        
        model = ResUNET_channel_attention(in_channels=config["model_params"]["in_channels"], out_channels=config["model_params"]["out_channels"],)
        model = nn.DataParallel(model)
        model = model.to(DEVICE)
        
        optimizer = Ranger(model.parameters(), lr=LEARNING_RATE)
        
        history = Fit(model=model,
                      train_loader=train_dl,
                      valid_loader=validation_dl,
                      device=DEVICE,
                      writer=writer,
                      dice_loss=dice_loss_fn,
                      ce_loss=CrossEntropyLoss_fn,
                      jaccard_loss=jaccard_loss_fn,
                      kl_divergence=None,
                      optimizer=optimizer,
                      epochs=EPOCHS,
                      )
        
         
        
        

if __name__ == "__main__":
    ## config dictionary for model training
    config = {
    "model_name": "ResUNET_channel_attention",
    "model_path": "./saved_models/",
    "data_path": "D:\\Saeed Ahmad Work\\MMCM_KD\\mmcm_kd\\BraTS_Dataset\\",
    "data_split_path": "./data_splits/",
    "writer_path": "./runs/",
    "batch_size": 2,
    "num_epochs": 100,
    "image_height": 128,
    "image_width": 128,
    "image_depth": 128,
    "model_params": {
        "in_channels": 3,
        "out_channels": 4,
        "dropout": 0.3,
        "attention": True,
        "attention_type": "channel",
        "learning_rate": 0.0001,
        },
    }

    #run the model   
    run(config=config)
