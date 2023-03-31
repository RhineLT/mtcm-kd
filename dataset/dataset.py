import os
from torch.utils.data import Dataset
from sklearn.model_selection import KFold, train_test_split
import nibabel as nib
import torch
import numpy as np
from torch.nn.functional import interpolate

class BraTS_Dataset(Dataset):
    def __init__(self, dataset_dir, data_dict, data_type, transform=None, target_transform=None):
        self.dataset_dir = dataset_dir
        self.transfrom = transform
        self.target_transform = target_transform
        self.samples = data_dict[f'{data_type}_samples']


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        voxels = []
        img_id = self.samples[index]

        for nii in [f'{self.dataset_dir}/{img_id}/{img_id}_{s_type}.nii.gz' for s_type in ["flair", "t1ce", "t2", "seg"]]: #"t1"
    
            image= nib.load(nii,).get_fdata()
            voxels.append(image)
        
        ## converting the data to numpy arrays
        volume = np.array(voxels[0:3])
        volume_mask = np.array(voxels[-1])

        ## converting the data to tensors
        image_volume = torch.FloatTensor(volume)
        image_mask = torch.Tensor(volume_mask)
        
        image_volume = image_volume[:,:,:,3:147]
        image_mask = image_mask[:,:, 3:147]

        image_mask[image_mask == 4] = 3

        if self.transfrom:
            image_volume = self.transfrom(image_volume)

        ## min max scaler for only non zero values
        #image_volume = image_volume - image_volume.min() / image_volume.max() - image_volume.min()
        if image_volume[image_volume != 0].min() != image_volume[image_volume != 0].max():
            image_volume[image_volume != 0] = (image_volume[image_volume != 0] - image_volume[image_volume != 0].min()) / (image_volume[image_volume != 0].max() - image_volume[image_volume != 0].min())
        
        
        if self.target_transform:  
            image_mask = image_mask.unsqueeze(0) 
            image_mask = self.target_transform(image_mask)
            image_mask = image_mask.squeeze(0)
        
        image_mask = image_mask.long()
        #image_volume = image_volume.type(torch.float16)

        return image_volume, image_mask



def spliting_data_5_folds(dataset_dir):
    '''
    This function split the dataset indices to five folds, and return the dictionary, which contains the indices 
    for five fold cross validation data, every fold have their corresponding train and validation indices.
    '''
    folds_data = []
    folders = os.listdir(dataset_dir)

    #train_valid_folders, test_folder, _ , _ = train_test_split(folders, folders, random_state=20, test_size=0.20)

    kfold = KFold(n_splits=5, shuffle=True, random_state=20)

    indices = kfold.split(folders, folders)

    for train_indices, valid_indices in indices:
        train_samples = [folders[index] for index in train_indices]
        valid_samples = [folders[index] for index in valid_indices]

        folds_data.append({
            "train_samples": train_samples,
            "valid_samples": valid_samples,
        })

    return folds_data

class reshape_3d(torch.nn.Module):
    """
    resize an 3d volume to a given shape
    inputs:
        volume: the input 3d volume, an nd array
        out_shape: the desired output shape, a list
        order: the order of interpolation
    outputs:
        out_volume: the reized 3d volume with given shape
    """
    def __init__(self, height, width, depth, mode='nearest'):
        super(reshape_3d, self).__init__()
        self.height = height
        self.width = width
        self.depth = depth
        self.mode = mode

    def forward(self, x):
        
        if len(x.shape) == 4:
            x = x.unsqueeze(0)
            x = interpolate(x,size=(self.height, self.width, self.depth), mode=self.mode, )
            x = x.squeeze(0)
        else:
            x = x.unsqueeze(0).unsqueeze(0)
            x = interpolate(x,size=(self.height, self.width, self.depth), mode=self.mode, )
            x = x.squeeze(0).squeeze(0)
        return x


def reshape_for_deep_supervision(y, dimentions):
    reshape = reshape_3d(dimentions, dimentions, dimentions)
    y =  y.float()
    y = reshape(y)
    y = y.long()
    return y
    



def test():
   
    folds_data = spliting_data_5_folds(dataset_dir="./BraTS_Dataset")
    test_ds = BraTS_Dataset(dataset_dir="./BraTS_Dataset", data_dict=folds_data[0], data_type='train')
    x, y = test_ds[1]

    y = reshape_for_deep_supervision(y, 64)
    
    print(f'y shape: {y.shape}')

if __name__ == "__main__":
    test()