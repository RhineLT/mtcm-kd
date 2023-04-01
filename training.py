import torch

from metrics import calculate_dice_score, calculate_hd95_multi_class, multiclass_dice_coeff
import json


def train_one_epoch(models, optimizers, loss_functions, lr_shedulars, train_loader, epoch, device, writer):
    
    
    """
    param: model: model to train
    param: optimizer: optimizer to use
    param: dice_loss: dice loss function
    param: jaccard_loss: jaccard loss function
    param: ce_loss: cross entropy loss function
    param: kl_divergence: kl divergence loss function
    param: train_loader: train loader
    param: epoch: current epoch
    param: device: device to use
    param: writer: tensorboard writer
    
    return: None
    
    Description: train one epoch
    
    """
    KL_Loss = False
    
    mean_loss = 0
    models['student_model'].train()
    
    for idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        output = models['student_model']((data )[:, 1, ...].unsqueeze(1))
        #teacher_output2 = t1_model(data[:, 0, ...].unsqueeze(1))
       # if idx % 50 == 0:
          #  t1_model.eval()
           # with torch.no_grad():
                #teacher_output = t1_model(data[:, 0, ...].unsqueeze(1))
               # kl_divergence_loss = torch.nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(output, dim=1), torch.softmax(teacher_output, dim=1))
               # KL_Loss = True
        #else: 
           # KL_Loss = False
        #loss = (dice_loss(target, output) + jaccard_loss(target, output) + ce_loss(output, target))/3.0
        loss = loss_functions['combination_loss'](target, output) #+ 0.01 * kl_divergence_loss) if KL_Loss else combination_loss(target, output)
       # teacher_loss = dice_loss(target, teacher_output2)
        
       # tm1_optimizer.zero_grad()
       #teacher_loss.backward()
        #tm1_optimizer.step()
        
        # zero the parameter gradients
        optimizers['student_optimizer'].zero_grad()
        
        # backward pass
        loss.backward()
        
        ## update weights
        optimizers['student_optimizer'].step()
        
        
        
        mean_loss += loss.detach().cpu().item()
        
        if idx % 50 == 0:
            print(f"Epoch: {epoch} | Batch: {idx} | Loss: {loss.item()}")
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + idx)
            preds = torch.softmax(output, dim=1)
            dice_dict = multiclass_dice_coeff(preds=preds, target=target)
            print(f"dice mean score: {dice_dict['mean']}")
            print(f"N-NE dice score: {dice_dict['N-NE']}")
            print(f"ED dice score: {dice_dict['ED']}")
            print(f"ET dice score: {dice_dict['ET']}")
            print(f"Whole tumor dice score: {dice_dict['whole_tumor']}")
            print(f"Tumor core dice score: {dice_dict['tumor_core']}")
            print("===========================================")
            
   # if epoch <= 6:
        ## update learning rate
        #print(f"Previous learning rate: {lr_shedulars['one_cycle'].get_lr()}")
        #lr_shedulars['one_cycle'].step()
        #print(f"Learning rate: {lr_shedulars['one_cycle'].get_lr()}")
        
    
    print("===========================================")
    return mean_loss / len(train_loader)


def validitation_loss(models, loss_functions, lr_shedulars, valid_loader, epoch, device, writer):
    """
    param: model: model to train
    param: dice_loss: dice loss function
    param: jaccard_loss: jaccard loss function
    param: ce_loss: cross entropy loss function
    param: kl_divergence: kl divergence loss function
    param: valid_loader: valid loader
    param: epoch: current epoch
    param: device: device to use
    param: writer: tensorboard writer
    
    return: None
    
    Description: calculate validitation loss
    
    """
    mean_loss = 0
    models['student_model'].eval()
    
    dice_dict = {}
    dice_dict["ED"] = 0
    dice_dict["ET"] = 0
    dice_dict["N-NE"] = 0
    dice_dict["mean"] = 0
    dice_dict["whole_tumor"] = 0
    dice_dict["tumor_core"] = 0
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(valid_loader):
            data = data.to(device)
            target = target.to(device)
            
            output = models['student_model']((data)[:, 1, ...].unsqueeze(1))
            
            #loss = (dice_loss(target, output) + jaccard_loss(target, output) + ce_loss(output, target))/3.0
            loss = loss_functions['combination_loss'](target, output)
            
            mean_loss += loss.detach().cpu().item()
            
          
            preds = torch.softmax(output, dim=1)
            temp_dice_dict = multiclass_dice_coeff(preds=preds, target=target)
            dice_dict['mean'] += temp_dice_dict['mean'].detach().cpu().item()
            dice_dict['N-NE'] += temp_dice_dict['N-NE'].detach().cpu().item()
            dice_dict['ED'] += temp_dice_dict['ED'].detach().cpu().item()
            dice_dict['ET'] += temp_dice_dict['ET'].detach().cpu().item()
            dice_dict['whole_tumor'] += temp_dice_dict['whole_tumor'].detach().cpu().item()
            dice_dict['tumor_core'] += temp_dice_dict['tumor_core'].detach().cpu().item()
            
        
        #if epoch >= 8:
            ## update learning rate
            #print("Previous learning rate: ", lr_shedulars['plateau'].optimizer.param_groups[0]['lr'])
           # lr_shedulars['plateau'].step(mean_loss / len(valid_loader))
           # print("Learning rate: ", lr_shedulars['plateau'].optimizer.param_groups[0]['lr'])
        
        
        dice_dict['mean'] /= len(valid_loader)
        dice_dict['N-NE'] /= len(valid_loader)
        dice_dict['ED'] /= len(valid_loader)
        dice_dict['ET'] /= len(valid_loader)
        dice_dict['whole_tumor'] /= len(valid_loader)
        dice_dict['tumor_core'] /= len(valid_loader)
        
        
        print(f"Epoch: {epoch} | Valid Loss: {mean_loss / len(valid_loader)}")
        print("mean loss: ", mean_loss / len(valid_loader))
        print(f"dice mean score: {dice_dict['mean']}")
        print(f"N-NE dice score: {dice_dict['N-NE']}")
        print(f"ED dice score: {dice_dict['ED']}")
        print(f"ET dice score: {dice_dict['ET']}")
        print(f"Whole tumor dice score: {dice_dict['whole_tumor']}")
        print(f"Tumor core dice score: {dice_dict['tumor_core']}")
        
        print("===========================================")
        
        
        writer.add_scalar("Loss/valid", mean_loss / len(valid_loader), epoch)
        
    return mean_loss / len(valid_loader), dice_dict



def Fit(models, optimizers, loss_functions, lr_schedulars, train_loader, valid_loader, epochs, device, writer, model_name, fold):
    """
    param: model: model to train
    param: optimizer: optimizer to use
    param: dice_loss: dice loss function
    param: jaccard_loss: jaccard loss function
    param: ce_loss: cross entropy loss function
    param: kl_divergence: kl divergence loss function
    param: train_loader: train loader
    param: valid_loader: valid loader
    param: epochs: number of epochs
    param: device: device to use
    param: writer: tensorboard writer
    
    return: None
    
    Description: train model
    
    """
    best_loss = 100000
    best_dice = 0
    
    ## results path
    results_path = "mmcm_kd\\results"
    models_path = "mmcm_kd\\saved_models\\"
    
    train_losses = []
    validitation_losses = []
    
    
    ### patience for early stopping
    patience = 5
    
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(models, optimizers, loss_functions, lr_schedulars, train_loader, epoch, device, writer)
        valid_loss, dice_dict = validitation_loss(models, loss_functions,lr_schedulars, valid_loader, epoch, device, writer,)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(models['student_model'].state_dict(), f"mmcm_kd\\saved_models\\best_loss_{fold}.pth")
        
        if dice_dict['mean'] > best_dice:
            best_dice = dice_dict['mean']
            torch.save(models['student_model'].state_dict(), f"mmcm_kd\\saved_models\\best__{fold}dice.pth")
            
        
        ## save the model 
        torch.save(models['student_model'].state_dict(), f"mmcm_kd\\saved_models\\{model_name}\\model_{fold}_{epoch}.pth")
        ## dump the dice dict to json file
        with open(f"mmcm_kd\\results\\{model_name}\\dice_dict_{fold}_{epoch}.json", "w") as f:
            json.dump(dice_dict, f)
        
        
        
        train_losses.append(train_loss)
        validitation_losses.append(valid_loss)
        
        ## early stopping condition for valid loss
        if epoch > patience:
            if validitation_losses[-patience] < validitation_losses[-1]:
                print("Early stopping")
                break
        
        
    history = {'epochs': epochs, 'train_loss': train_losses, 'valid_loss': validitation_losses}
        
    writer.close()
    print("Training Finished")
    return history