import os

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


import torch
import torch.nn.functional as F
    

__all__ = ["save_volume", "predict_and_save_volume", "convert_to_one_hot", "convert_one_hot_to_label_encoding"]

### saving the 3d volumes in .nii.gz format
def save_volume(volume, save_path):
    """
    Input Parameters:
        volume: volume to be saved
        save_path: path where the volume is to be saved
        
    Output:
        saves the volume in the specified path
    
    Description:
        This function saves the volume in .nii.gz format
        
    """
    img = nib.Nifti1Image(volume, np.eye(4))
    nib.save(img, save_path)
     

def convert_to_one_hot(x, models, modality=1):
    """
    Input Parameters:
        x: input volume
        model: models to be used for prediction
    
    Output:
        one_hot_volume: one hot encoded volume
    
    Description:
        This function first predicts the volume using the model and then converts the predicted volume to one hot encoding
    """
   
    with torch.no_grad():
        outputs = []  
        for fold in range(5):
            models[fold].eval()
            
        for fold in range(5):
                outputs.append(models[fold](x[:, modality, ...].unsqueeze(1)))
        final_output = torch.mean(torch.stack(outputs), dim=0)
    
    
        preds = torch.softmax(final_output, dim=1)
        one_hot_volume = F.one_hot(preds.argmax(1), 4).permute(0, 4, 1, 2, 3)
    return one_hot_volume


## convert one hot to label encoding
def convert_one_hot_to_label_encoding(one_hot_volume):
    """
    Input Parameters:
        one_hot_volume: one hot encoded volume
    
    Output:
        label_encoding_ET: label encoding of the enhancing tumor
        label_encoding_WT: label encoding of the whole tumor
        label_encoding_TC: label encoding of the tumor core
    
    Description:
        This function converts the one hot encoded volume to label encoding
    """
    label_encoding_ET = np.zeros((one_hot_volume.shape[1], one_hot_volume.shape[2], one_hot_volume.shape[3]))
    label_encoding_WT = np.zeros((one_hot_volume.shape[1], one_hot_volume.shape[2], one_hot_volume.shape[3]))
    label_encoding_TC = np.zeros((one_hot_volume.shape[1], one_hot_volume.shape[2], one_hot_volume.shape[3]))
    
    for i in range(one_hot_volume.shape[1]):
        for j in range(one_hot_volume.shape[2]):
            for k in range(one_hot_volume.shape[3]):
                
                if one_hot_volume[1, i, j, k] == 1  or one_hot_volume[3, i, j, k] == 1: ## Tumor core 
                    label_encoding_TC[i, j, k] = 1
                    
                if one_hot_volume[1, i, j, k] == 1  or one_hot_volume[2, i, j, k] == 1 or one_hot_volume[3, i, j, k] == 1 :  ## Whole tumor
                    label_encoding_WT[i, j, k] = 1
                    
                if one_hot_volume[3, i, j, k] == 1: ##  Enhancing tumor
                    label_encoding_ET[i, j, k] = 1
                    
    return label_encoding_ET, label_encoding_WT,  label_encoding_TC


### convert ground truth to separate regions for visualization
def convert_gt_regions(y):
    """
    Input Parameters:
        y: ground truth volume
    
    Output:
        label_encoding_ET: label encoding of the enhancing tumor
        label_encoding_WT: label encoding of the whole tumor
        label_encoding_TC: label encoding of the tumor core
    
    Description:
        This function converts the ground truth volume to label encoding
    """
    label_encoding_ET = np.zeros((y.shape[0], y.shape[1], y.shape[2]))
    label_encoding_WT = np.zeros((y.shape[0], y.shape[1], y.shape[2]))
    label_encoding_TC = np.zeros((y.shape[0], y.shape[1], y.shape[2]))
    
    
    label_encoding_ET[y == 3] = 1
    label_encoding_TC = np.where((y== 1) | (y==3), 1, 0) 
    label_encoding_TC = np.where((y==1) | (y==2) | (y==3), 1, 0)
                        
    return label_encoding_ET, label_encoding_WT,  label_encoding_TC

### predicting_and_saving_the_volumes
def predict_and_save_volume(models, sample, test_batch, model_name, device, modality=0):
    """
    Input Parameters: 
        model: model to be used for prediction
        sample: sample number to be predicted
        test_dl: test dataloader
        model_name: name of the model
        device: device to be used for prediction
        modality: modality of the input image 
    
    Output:
        saves the predicted volume, input volume and ground truth volume in .nii.gz format
    
    Description:
        This function predicts the volume for the given sample number and saves the predicted volume, input volume and ground truth volume in .nii.gz format

    """
    
    x, y = test_batch

    x = x.to(device)
    y = y.to(device)
        
    one_hot = convert_to_one_hot(models=models, x=x, modality=modality)
    
    
    pred_ET, pred_WT, pred_TC = convert_one_hot_to_label_encoding(one_hot[0,:,:,:].detach().cpu().numpy())
    y_ET, y_WT, y_TC = convert_gt_regions(y[0,:,:,:].detach().cpu().numpy())
    
    
    if not os.path.exists('model_predictions'):
        os.makedirs('model_predictions')
        
    prediction_dir = "model_predictions"
    
    if not os.path.exists(os.path.join(prediction_dir, str(sample))):
        os.makedirs(os.path.join(prediction_dir, str(sample)))
    
    ## save the predicted volume 
    save_volume(pred_ET, os.path.join(prediction_dir, str(sample), "prediction_ET.nii.gz"))
    save_volume(pred_WT, os.path.join(prediction_dir, str(sample), "prediction_WT.nii.gz"))
    save_volume(pred_TC, os.path.join(prediction_dir, str(sample), "prediction_TC.nii.gz"))
    
    save_volume(y_ET, os.path.join(prediction_dir, str(sample), "y_ET.nii.gz"))
    save_volume(y_WT, os.path.join(prediction_dir, str(sample), "y_WT.nii.gz"))
    save_volume(y_TC, os.path.join(prediction_dir, str(sample), "y_TC.nii.gz"))
    
    save_volume(x[0, modality, ...].detach().cpu().numpy(), os.path.join(prediction_dir, str(sample), "x.nii.gz"))