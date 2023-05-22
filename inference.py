import os

import torch
from torchvision import transforms as t
import torch.nn as nn
import json



from models import ResUNET_channel_attention
from optimizer import Ranger
from dataset import get_test_loaders, reshape_3d
from metrics import calculate_dice_score, calculate_hd95_multi_class, save_history, multiclass_dice_coeff


def inference(config, data_dict, dataset_dir):

    ## device configuration
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    ## settings
    EPOCHS = config["num_epochs"]
    HEIGHT = config["image_height"]
    WEIGHT = config["image_width"]
    DEPTH = config["image_depth"]

    BATCH_SIZE = 3 #config["batch_size"]
    LEARNING_RATE = config["model_params"]["learning_rate"]



    ## data loading and processing
    reshape = reshape_3d(HEIGHT, WEIGHT, DEPTH)
    def reshape_volume(x): return reshape(x)
    
    

    
    ## transforms
    general_transforms = t.Compose([
       t.Lambda(reshape_volume),
    ])
    
        
    ## get the data loaders
    test_dl = get_test_loaders(
        dataset_dir = dataset_dir,
        batch_size = BATCH_SIZE,
        data_dict = data_dict,
        test_images_transform = general_transforms,
        test_masks_transform = general_transforms,
    )
        
        
    ## get the gpu devices
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
 
   
    student_models = []
    
    ## define the model
    for fold in range(5):
        model = ResUNET_channel_attention(in_channels=config["model_params"]["in_channels"], out_channels=config["model_params"]["out_channels"],)
        model = nn.DataParallel(model, device_ids=[0])
        model = model.to(devices[0])
        student_models.append(model)
    
    ## load the models
    for fold in range(5):
        model_path = os.path.join(config["model_path"], config["model_name"], f"best_loss_{fold}.pth")
        student_models[fold].load_state_dict(torch.load(model_path))
    
    
    ### test the model
    dice_dict = test_models(models=student_models, test_loader=test_dl, device=devices)
    
    
    
        
def read_data(dataset_dir):
    """
    parameters:
        dataset_dir: the directory of the dataset
        
    return:
        data: a list of dictionary, each dictionary contains the information of the dataset
    """
    data_samples = os.listdir(dataset_dir)
    data = []
    data.append({"test_samples": data_samples,})
    return data   


def test_models(models, test_loader, device):
    """
    param:
        models: a list of models for testing
        test_loader: the data loader for testing
        device: the device for testing the model
    
    return: None
    
    Description: calculate the dice score for the testing data
    
    """
   
    for fold in range(5):
        models[fold].eval()
    
    dice_dict = {}
    dice_dict["ED"] = 0
    dice_dict["ET"] = 0
    dice_dict["N-NE"] = 0
    dice_dict["mean"] = 0
    dice_dict["whole_tumor"] = 0
    dice_dict["tumor_core"] = 0
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data = data.to(device[0])
            target = target.to(device[0])
            
            
            ## torch list to tensor
            outputs = []
            
            for fold in range(5):
                outputs.append(models[fold](data[:, 1, ...].unsqueeze(1)))
            
            final_output = torch.mean(torch.stack(outputs), dim=0)

            
          
            preds = torch.softmax(final_output, dim=1)
            temp_dice_dict = multiclass_dice_coeff(preds=preds, target=target)
            dice_dict['mean'] += temp_dice_dict['mean'].detach().cpu().item()
            dice_dict['N-NE'] += temp_dice_dict['N-NE'].detach().cpu().item()
            dice_dict['ED'] += temp_dice_dict['ED'].detach().cpu().item()
            dice_dict['ET'] += temp_dice_dict['ET'].detach().cpu().item()
            dice_dict['whole_tumor'] += temp_dice_dict['whole_tumor'].detach().cpu().item()
            dice_dict['tumor_core'] += temp_dice_dict['tumor_core'].detach().cpu().item()
            
        

        
        dice_dict['mean'] /= len(test_loader)
        dice_dict['N-NE'] /= len(test_loader)
        dice_dict['ED'] /= len(test_loader)
        dice_dict['ET'] /= len(test_loader)
        dice_dict['whole_tumor'] /= len(test_loader)
        dice_dict['tumor_core'] /= len(test_loader)
        
        print("===========================================")
        print(f"dice mean score: {dice_dict['mean']}")
        print(f"N-NE dice score: {dice_dict['N-NE']}")
        print(f"ED dice score: {dice_dict['ED']}")
        print(f"ET dice score: {dice_dict['ET']}")
        print(f"Whole tumor dice score: {dice_dict['whole_tumor']}")
        print(f"Tumor core dice score: {dice_dict['tumor_core']}")
        
        print("===========================================")
        
        ## save the dice dict
        dataset_name = "BraTS_2020"
        with open(os.path.join("results", "test_results", config["model_name"],  f"{dataset_name}_dice_dict.json"), "w") as f:
            json.dump(dice_dict, f)
        
        
    return dice_dict






if __name__ == "__main__":
   
    dataset_dir = "BraTS_2020/MICCAI_BraTS2020_TrainingData"
    config = json.load(open("config.json"))
    data = read_data(dataset_dir= dataset_dir)
    inference(config=config, data_dict=data[0], dataset_dir=dataset_dir)
    