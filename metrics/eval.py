import json
import torch
from torch import Tensor
import torch.nn.functional as F
from  torch.cuda.amp import autocast
from fastai.callbacks import *
from fastai.vision import *

from medpy.metric import hd95

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(preds: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes

    target = F.one_hot(target, 4).permute(0,4,1,2,3).float()
    input = F.one_hot(preds.argmax(1), 4).permute(0,4,1,2,3).float()

    input = input[:, 1:, :, :, :]
    target = target[:, 1:, :, :, :]

    assert input.size() == target.size()
    dice = 0
    all_dice_score = []
    for channel in range(input.shape[1]):
        classwise_dice = dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
        all_dice_score.append(classwise_dice)
        dice += classwise_dice
        
    
    ## whole tumor
    def change_to_one(input):
        input[:, 0, ...] = torch.where((input[:,0, ...] > 0) |(input[:,1, ...] > 0) | (input[:,2, ...] > 0), torch.tensor([1], device="cuda:0"), torch.tensor([0], device="cuda:0"))
        return input[:, 0, ...]
    def change_to_two(input):
        input[:, 1, ...] = torch.where((input[:,1, ...] > 0) | (input[:,2, ...] > 0), torch.tensor([1], device="cuda:0"), torch.tensor([0], device="cuda:0"))
        return input[:, 1, ...]
    
    
    tumor_core_pred = change_to_two(input)
    tumor_core_target = change_to_two(target)
    tumor_core = dice_coeff(tumor_core_pred, tumor_core_target, reduce_batch_first, epsilon)
    
    whole_tumor_pred = change_to_one(input)
    whole_tumor_target = change_to_one(target)
    whole_tumor = dice_coeff(whole_tumor_pred, whole_tumor_target, reduce_batch_first, epsilon)
    
    
    
    
    
    dice_dict = {}
    dice_dict['mean'] = dice / input.shape[1]
    dice_dict['N-NE'] = all_dice_score[0]
    dice_dict['ED'] = all_dice_score[1]
    dice_dict['ET'] = all_dice_score[2]
    dice_dict['whole_tumor'] = whole_tumor
    dice_dict['tumor_core'] = tumor_core

    return  dice_dict
    


def calculate_dice_score(model, loader, device, save_results=False, epoch=None, data=None, model_name=None):
    
    dice_dict = {}
    dice_dict['mean'] = 0.0
    dice_dict['N-NE'] = 0.0
    dice_dict['ED'] = 0.0
    dice_dict['ET'] = 0.0
    dice_dict['whole_tumor'] = 0.0
    dice_dict['tumor_core'] = 0.0
    
    model.eval()
    with torch.no_grad():
        for  x, y in loader:
            x = x.to(device)
            y = y.to(device)
  
            output = model(x)
            
            preds = torch.softmax(output, dim=1)
            batch_dict = multiclass_dice_coeff(preds=preds, target=y)
            dice_dict['mean'] += batch_dict['mean']
            dice_dict['N-NE'] += batch_dict['N-NE']
            dice_dict['ED'] += batch_dict['ED']
            dice_dict['ET'] += batch_dict['ET']
            dice_dict['whole_tumor'] += batch_dict['whole_tumor']
            dice_dict['tumor_core'] += batch_dict['tumor_core']
    

    dice_dict['mean'] /= len(loader)
    dice_dict['N-NE'] /= len(loader)
    dice_dict['ED'] /= len(loader)
    dice_dict['ET'] /= len(loader)  
    dice_dict['whole_tumor'] /= len(loader)
    dice_dict['tumor_core'] /= len(loader)

    dice_dict['mean'] = dice_dict['mean'].detach().cpu().item()
    dice_dict['N-NE'] = dice_dict['N-NE'].detach().cpu().item()
    dice_dict['ED'] = dice_dict['ED'].detach().cpu().item()
    dice_dict['ET'] = dice_dict['ET'].detach().cpu().item()
    dice_dict['whole_tumor'] = dice_dict['whole_tumor'].detach().cpu().item()
    dice_dict['tumor_core'] = dice_dict['tumor_core'].detach().cpu().item()

    print(f"dice mean score: {dice_dict['mean']}")
    print(f"N-NE dice score: {dice_dict['N-NE']}")
    print(f"ED dice score: {dice_dict['ED']}")
    print(f"ET dice score: {dice_dict['ET']}")
    print(f"whole tumor dice score: {dice_dict['whole_tumor']}")
    print(f"tumor core dice score: {dice_dict['tumor_core']}")

    if save_results:
        json_file = json.dumps(dice_dict)
        f = open(f"./results/{model_name}/epoch_{epoch}_{data}.json", "w")
        f.write(json_file)
        f.close()

    return dice_dict


def calculate_hd95_multi_class(preds, target, spacing=None, connectivity=1):
    hd95_dict = {}
    hd95_dict['mean'] = 0.0
    hd95_dict['N-NE'] = 0.0
    hd95_dict['ED'] = 0.0
    hd95_dict['ET'] = 0.0
    
    target = F.one_hot(target, 4).permute(0,4,1,2,3).float()
    preds = F.one_hot(preds.argmax(1), 4).permute(0,4,1,2,3).float()

    preds = preds[:, 1:, :, :, :]
    target = target[:, 1:, :, :, :]

    assert preds.size() == target.size()

    for i in range(preds.shape[0]):
        batch_hd95_dict = hd95(preds[i, ...], target[i, ...], spacing, connectivity)
        hd95_dict['mean'] += batch_hd95_dict['mean']
        hd95_dict['N-NE'] += batch_hd95_dict['N-NE']
        hd95_dict['ED'] += batch_hd95_dict['ED']
        hd95_dict['ET'] += batch_hd95_dict['ET']

    hd95_dict['mean'] /= preds.shape[0]
    hd95_dict['N-NE'] /= preds.shape[0]
    hd95_dict['ED'] /= preds.shape[0]
    hd95_dict['ET'] /= preds.shape[0]

    return hd95_dict




def save_history(history, model_name, epochs,fold_no):
    json_file = json.dumps(history)
    f = open(f"{model_name}/history_epoch_{epochs}_fold_no_{fold_no}.json", "w")
    f.write(json_file)
    f.close()