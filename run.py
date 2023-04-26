
import os

import torch
from torchvision import transforms as t
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
#import torchio.transforms as tio

from training import  Fit
from models import ResUNET_channel_attention
from models.cooperative_learning import Cooperative_Learning_Module
from loss_functions import dice_loss, jaccard_loss, CrossEntropyLoss, KL_divergence, combination_loss
from optimizer import Ranger
from dataset import  get_loaders, spliting_data_5_folds, reshape_for_deep_supervision, reshape_3d
from metrics import calculate_dice_score, calculate_hd95_multi_class, save_history
from utils import initialize_weights




def run(config):

    ## device configuration
    
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
    # Define the transforms
   # rotation_scale_transform = t.Compose([
       # t.RandomRotation(degrees=15),
       # t.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.2)),
    #])

    #elastic_transform = tio.Compose([
      #  tio.RandomElasticDeformation(num_control_points=(7, 7, 7), max_displacement=(5, 5, 5)),
   # ])

    #brightness_transform = t.Lambda(lambda x: torch.clamp(x + 0.2 * torch.randn_like(x), 0, 1))

    #gamma_transform = t.Lambda(lambda x: torch.pow(x, 0.7))
    
    ## transforms
    general_transforms = t.Compose([
       t.Lambda(reshape_volume),
    ])
    
    ## transforms
    train_transforms = t.Compose([
       t.Lambda(reshape_volume),
       ## augmentation for 3d volume  data
        #rotation_scale_transform,
       # elastic_transform,
       # brightness_transform,
       # gamma_transform,
       
    ])
    

    ## spliting data into 5 folds holdout validation
    ## check for the folds_data.json file, if not exist, create it
    if not os.path.exists(config["data_split_path"]):
        os.makedirs(config["data_split_path"])
        data_split = spliting_data_5_folds(dataset_dir=config["data_path"])
        
        ## convert the dictionary to json file
        dict_to_json = json.dumps(data_split)
        
        ## save the json file
        with open(config["data_split_path"] + "//folds_data.json", "w") as file:
            file.write(dict_to_json)
    else:
        with open(config["data_split_path"] + "//folds_data.json", "r") as file:
            data_split = json.load(file)
            

    dice_loss_fn = dice_loss
    jaccard_loss_fn = jaccard_loss
    CrossEntropyLoss_fn = CrossEntropyLoss()
    combination_loss_fn = combination_loss
    KL_divergence_fn = KL_divergence
    
    
    for fold_index in range(0, 5):
        
         ## get the data loaders
        train_dl, validation_dl = get_loaders(
            dataset_dir=config["data_path"],
            batch_size=BATCH_SIZE,
            data_dict=data_split[fold_index],
            train_images_transform = train_transforms,
            train_masks_transform = train_transforms,
            valid_images_transform = general_transforms,
            valid_masks_transform = general_transforms,
        
            )
        
        
        ## get the gpu devices
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
 
        ## model configuration
        writer = SummaryWriter(log_dir=config["writer_path"] + "//" + config["model_name"] + "//" +f"fold_{fold_index}")
        
        student_model = ResUNET_channel_attention(in_channels=config["model_params"]["in_channels"], out_channels=config["model_params"]["out_channels"],)
        student_model = nn.DataParallel(student_model, device_ids=[0])
        student_model = student_model.to(devices[0])
        student_model.apply(initialize_weights)
        
        teacher_model1 = ResUNET_channel_attention(in_channels=config["model_params"]["in_channels"], out_channels=config["model_params"]["out_channels"],)
        teacher_model1 = nn.DataParallel(teacher_model1, device_ids=[1])
        teacher_model1 = teacher_model1.to(devices[1])
        
        teacher_model2 = ResUNET_channel_attention(in_channels=config["model_params"]["in_channels"], out_channels=config["model_params"]["out_channels"],)
        teacher_model2 = nn.DataParallel(teacher_model2, device_ids=[1])
        teacher_model2 = teacher_model2.to(devices[1])
        
        teacher_model3 = ResUNET_channel_attention(in_channels=config["model_params"]["in_channels"], out_channels=config["model_params"]["out_channels"],)
        teacher_model3 = nn.DataParallel(teacher_model3, device_ids=[1])
        teacher_model3 = teacher_model3.to(devices[1])
        
        Cooperative_learning1 = Cooperative_Learning_Module(in_channels=64, out_channels=4)
        Cooperative_learning1 = nn.DataParallel(Cooperative_learning1, device_ids=[1])
        Cooperative_learning1 = Cooperative_learning1.to(devices[1])
        
        
        Cooperative_learning2 = Cooperative_Learning_Module(in_channels=32, out_channels=4)
        Cooperative_learning2 = nn.DataParallel(Cooperative_learning2, device_ids=[1])
        Cooperative_learning2 = Cooperative_learning2.to(devices[1])
        
        
        sm_optimizer = optim.Adam(student_model.parameters(), lr=LEARNING_RATE) 
        #tm_optimizer1 = optim.Adam(teacher_model1.parameters(), lr=LEARNING_RATE)   
        #tm_optimizer2 = optim.Adam(teacher_model2.parameters(), lr=LEARNING_RATE)   
        #tm_optimizer3 = optim.Adam(teacher_model3.parameters(), lr=LEARNING_RATE)
        
        #cooperative_optimizer = optim.Adam(Cooperative_learning1.parameters(), lr=LEARNING_RATE)
        generalized_optimizer = optim.Adam(list(teacher_model1.parameters()) +
                                           list(teacher_model2.parameters()) + list(teacher_model3.parameters()) + 
                                           list(Cooperative_learning1.parameters(), list(Cooperative_learning2.parameters())), lr=LEARNING_RATE)
       
        ### learning schedulars 
        lr_scheduler_one_cycle = OneCycleLR(sm_optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_dl), epochs=EPOCHS)
        lr_scheduler_plateau = ReduceLROnPlateau(sm_optimizer, mode="min", factor=0.1, patience=5, verbose=True)
        
        
        models = {"student_model": student_model, "teacher_model1": teacher_model1, "teacher_model2": teacher_model2, "teacher_model3": teacher_model3,
                  "Cooperative_learning1": Cooperative_learning1 , "Cooperative_learning2": Cooperative_learning2}
        optimizers = {"student_optimizer": sm_optimizer, "teacher_optimizer1": None, "teacher_optimizer2": None, 
                      "teacher_optimizer3": None, "cooperative_optimizer": None, "generalized_optimizer": generalized_optimizer}
        loss_functions = {"dice_loss": dice_loss_fn, "jaccard_loss": jaccard_loss_fn, "cross_entropy_loss": CrossEntropyLoss_fn, "combination_loss": combination_loss_fn,
                          "kl_div_loss": KL_divergence_fn}
        lr_schedulars = {"one_cycle": lr_scheduler_one_cycle, "plateau": lr_scheduler_plateau}
        
        history = Fit(models= models,
                      optimizers= optimizers,
                      loss_functions= loss_functions,
                      lr_schedulars= lr_schedulars,
                      train_loader=train_dl,
                      valid_loader=validation_dl,
                      device=devices,
                      writer=writer,
                      epochs=EPOCHS,
                      model_name=config["model_name"],
                      fold=fold_index,
                      )
        
        save_history(history, config["results_path"] + "/"+ config["model_name"] , epochs=EPOCHS, fold_no=fold_index)
        
        ## 
        
    

if __name__ == "__main__":
    ## load config file
    config = json.load(open("config.json"))
    #run the model   
    run(config=config)
