import os
import json

import torch
from torchvision import transforms as t
import torch.nn as nn


from models import ResUNET_channel_attention
from dataset import get_test_loaders, reshape_3d, read_test_data
from metrics import multiclass_dice_coeff

from visualization import save_volume, convert_to_one_hot, convert_one_hot_to_label_encoding, predict_and_save_volume


"""
The purpose of this file is to predict some volumes and save them in .nii.gz format for visualization purposes
We use a tool called NIFTI 3D Visualizer to visualize the volumes (link: https://github.com/adamkwolf/3d-nii-visualizer)
"""


def visualize(config, data_dict, dataset_dir, testONT1ce=True):
    
    ## device configuration
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ## get the gpu devices 
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    
    ## model configuration
    HEIGHT = config["image_height"]
    WEIGHT = config["image_width"]
    DEPTH = config["image_depth"]
    
    BATCH_SIZE = config["batch_size"]
    
    ## dataset transforms
    reshape = reshape_3d(HEIGHT, WEIGHT, DEPTH)
    def reshape_volume(x): return reshape(x)
    
    general_transforms = t.Compose([ t.Lambda(reshape_volume), ])
    
    ## get the test data loader 
    test_dl, test_ds = get_test_loaders(
        dataset_dir = dataset_dir,
        batch_size = BATCH_SIZE,
        data_dict = data_dict,
        test_images_transform = general_transforms,
        test_masks_transform = general_transforms,
    )
    
    student_models = []
    
    ## define the model
    for fold in range(5):
        model = ResUNET_channel_attention(in_channels=config["model_params"]["in_channels"], out_channels=config["model_params"]["out_channels"],)
        model = nn.DataParallel(model)
        model = model.to(devices[0])
        student_models.append(model)
    
    ## load the models
    for fold in range(5):
        model_path = os.path.join(config["model_path"], config["model_name"], f"best_loss_{fold}.pth")
        student_models[fold].load_state_dict(torch.load(model_path))
        
        
    #performance dictionary
    #overall_performance = []
    
    
    ## reading the json file (performance.json) to get the indices of the samples to be visualized
    overall_performance = json.load(open("performance.json"))
    
    print(overall_performance[0].keys()) 
    
    ## visualize the volumes for worst 15 samples 
    for idx in range(10):
        sample = overall_performance[idx]["sample"]
        print(sample)
        x, y = test_ds[sample]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        
        x = x.to(devices[0])
        y = y.to(devices[0])
        
        ## predict the volume and save it in .nii.gz format
        predict_and_save_volume(models=student_models, sample=sample, test_batch=(x, y), model_name=config["model_name"], device=devices[0], modality=1)
       
    ## test the model on the test data
   # for idx, (x, y) in enumerate(test_dl):
       # x = x.to(devices[0])
       # y = y.to(devices[0])
        #dice_dict = test_one_batch(student_models, x, y, modality=1)
       # temp_dict = {
           # "sample": idx,
          #  "WT": dice_dict["whole_tumor"],
           # "TC": dice_dict["tumor_core"],
           # "ET": dice_dict["ET"],
           # "mean": dice_dict["mean"],
       # }
       # overall_performance.append(temp_dict)
    
    
    ## sort the performance dictionary on the mean dice score of all the samples
    #overall_performance = sorted(overall_performance, key=lambda i: i['mean'], reverse=True)
    
   # with open("performance.json", "w") as f:
       # json.dump(overall_performance, f)


def test_one_batch(models, x, y, modality=1):
    """
    param:
        models: a list of models for testing
        x: input data
        y: ground truth
        modality: modality of the input data
    
    return: dice_dict for the batch
    
    Description: This function tests one batch of data and returns the dice_dict for the batch
    """
   
    for fold in range(5):
        models[fold].eval()
    
    dice_dict = {}
    dice_dict["ET"] = 0
    dice_dict["whole_tumor"] = 0
    dice_dict["tumor_core"] = 0
    dice_dict["mean"] = 0
        
    with torch.no_grad():
                
        ## torch list to tensor
        outputs = []
            
        for fold in range(5):
            outputs.append(models[fold](x[:, modality, ...].unsqueeze(1)))
            
        final_output = torch.mean(torch.stack(outputs), dim=0)
          
    preds = torch.softmax(final_output, dim=1)
    temp_dice_dict = multiclass_dice_coeff(preds=preds, target=y)
                    
    dice_dict['ET'] = temp_dice_dict['ET'].detach().cpu().item()
    dice_dict['whole_tumor'] = temp_dice_dict['whole_tumor'].detach().cpu().item()
    dice_dict['tumor_core'] = temp_dice_dict['tumor_core'].detach().cpu().item()
    dice_dict['mean'] = (dice_dict['ET'] + dice_dict['whole_tumor'] + dice_dict['tumor_core']) / 3.0
    
    return dice_dict
    
    
if __name__ == "__main__":
    dataset_dir = "BraTS_2020/MICCAI_BraTS2020_TrainingData"
    config = json.load(open("config.json"))
    data_dict = read_test_data(dataset_dir)
    
    ## visualize the volumes
    visualize(config=config, data_dict=data_dict[0], dataset_dir=dataset_dir, testONT1ce=True)