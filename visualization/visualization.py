import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
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
     

def convert_to_one_hot(x, model):
    """
    Input Parameters:
        x: input volume
        model: model to be used for prediction
    
    Output:
        one_hot_volume: one hot encoded volume
    
    Description:
        This function first predicts the volume using the model and then converts the predicted volume to one hot encoding
    """
    output = model(x.unsqueeze(1))
    preds = torch.softmax(output, dim=1)
    one_hot_volume = F.one_hot(preds.argmax(1), 4).permute(0, 4, 1, 2, 3)
    return one_hot_volume


## convert one hot to label encoding
def convert_one_hot_to_label_encoding(one_hot_volume):
    """
    Input Parameters:
        one_hot_volume: one hot encoded volume
    
    Output:
        label_encoding_volume: label encoding of the one hot encoded volume
    
    Description:
        This function converts the one hot encoded volume to label encoding
    """
    label_encoding_volume = np.zeros((one_hot_volume.shape[1], one_hot_volume.shape[2], one_hot_volume.shape[3]))
    for i in range(one_hot_volume.shape[1]):
        for j in range(one_hot_volume.shape[2]):
            for k in range(one_hot_volume.shape[3]):
                if one_hot_volume[0, i, j, k] == 1:
                    label_encoding_volume[i, j, k] = 0
                elif one_hot_volume[1, i, j, k] == 1:
                    label_encoding_volume[i, j, k] = 1
                elif one_hot_volume[2, i, j, k] == 1:
                    label_encoding_volume[i, j, k] = 2
                elif one_hot_volume[3, i, j, k] == 1:
                    label_encoding_volume[i, j, k] = 3
    return label_encoding_volume




### predicting_and_saving_the_volumes
def predict_and_save_volume(model, sample, test_batch, model_name, device, x_modality=0):
    """
    Input Parameters: 
        model: model to be used for prediction
        sample: sample number to be predicted
        test_dl: test dataloader
        model_name: name of the model
        device: device to be used for prediction
        x_modality: modality of the input image to be saved
    
    Output:
        saves the predicted volume, input volume and ground truth volume in .nii.gz format
    
    Description:
        This function predicts the volume for the given sample number and saves the predicted volume, input volume and ground truth volume in .nii.gz format

    """
    
    x, y = test_batch

    x = x.to(device)
    y = y.to(device)
        
    one_hot = convert_to_one_hot(model=model, x=x)
    label_encoding = convert_one_hot_to_label_encoding(one_hot[0,:,:,:].detach().cpu().numpy())
   
   
    save_volume(label_encoding, f'{model_name}_{sample}_prediction.nii.gz')
    save_volume(x[0,x_modality,:,:].detach().cpu().numpy(), f'{model_name}_{sample}_x.nii.gz')
    save_volume(y[0,:,:,:].detach().cpu().numpy(), f'{model_name}_{sample}_y.nii.gz')