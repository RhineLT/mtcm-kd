from .dataset import BraTS_Dataset
from torch.utils.data import DataLoader

__all__ = ["get_loaders", "get_test_loaders"]

def get_loaders(
    dataset_dir,
    batch_size,
    data_dict,
    train_images_transform ,
    train_masks_transform,
    valid_images_transform,
    valid_masks_transform,
   # test_images_transform,
   # test_masks_transform,
    ):

    train_ds = BraTS_Dataset(
        dataset_dir= dataset_dir, 
        data_dict=data_dict,
        data_type='train',
        transform = train_images_transform, 
        target_transform = train_masks_transform,
        )

    validation_ds = BraTS_Dataset(
        dataset_dir= dataset_dir, 
        data_dict=data_dict,
        data_type='valid',
        transform = valid_images_transform, 
        target_transform = valid_masks_transform)

    #test_ds = BraTS_Dataset(
        #dataset_dir= dataset_dir, 
        #data_dict=data_dict,
        #data_type='test',
        #transform = test_images_transform, 
        #target_transform = test_masks_transform)
    
    
    train_dl = DataLoader(
        dataset = train_ds,
        batch_size = batch_size,
        shuffle = True, 
        pin_memory=True,
        )
    
    validation_dl = DataLoader(
        dataset = validation_ds,
        batch_size = batch_size,
        shuffle = False, 
        pin_memory=True,
        )
    
    #test_dl = DataLoader(
        #dataset = test_ds,
        #batch_size = batch_size,
       # shuffle = False, 
       # pin_memory=True,
       # )

    return train_dl, validation_dl #, test_dl


def get_test_loaders(
    dataset_dir,
    batch_size,
    data_dict,
    test_images_transform,
    test_masks_transform,
    ):


    test_ds = BraTS_Dataset(
        dataset_dir= dataset_dir, 
        data_dict=data_dict,
        data_type='test',
        transform = test_images_transform, 
        target_transform = test_masks_transform)
    
    test_dl = DataLoader(
        dataset = test_ds,
        batch_size = batch_size,
        shuffle = False, 
        pin_memory=True,
        )

    return  test_dl