import torch
import torch.nn as nn

from metrics import calculate_dice_score, calculate_hd95_multi_class, multiclass_dice_coeff
import json



# Define the KL divergence loss with temperature
def kd_loss(outputs, teacher_outputs, T):
    soft_outputs = nn.functional.softmax(outputs / T, dim=1)
    soft_teacher_outputs = nn.functional.softmax(teacher_outputs / T, dim=1)
    return T * nn.KLDivLoss(reduction='batchmean')(soft_outputs.log(), soft_teacher_outputs.detach())


def calculate_wt_dice_for_teacher_models(t1_preds, t2_preds, t3_preds, target, device):
    """
    param: t1_preds: predictions of teacher model 1
    param: t2_preds: predictions of teacher model 2
    param: t3_preds: predictions of teacher model 3
    param: target: target
    param: device: device to use
    
    return: weights: weights for teacher loss
    
    Description: calculate weights for teacher loss
    """
    t1_preds = t1_preds.to(device).detach()
    t2_preds = t2_preds.to(device).detach()
    t3_preds = t3_preds.to(device).detach()
    
    preds = torch.softmax(t1_preds, dim=1)
    t1_dice_dict = multiclass_dice_coeff(preds=preds, target=target)
    
    preds = torch.softmax(t2_preds, dim=1)
    t2_dice_dict = multiclass_dice_coeff(preds=preds, target=target)
    
    preds = torch.softmax(t3_preds, dim=1)
    t3_dice_dict = multiclass_dice_coeff(preds=preds, target=target)
    
    wt_dice_scores = [t1_dice_dict['whole_tumor'], t2_dice_dict['whole_tumor'], t3_dice_dict['whole_tumor']]
    
    return wt_dice_scores


def performance_base_weight_calculation(t1_wt_score, t2_wt_score, t3_wt_score, total_weight=0.10):
    """
    param: t1_wt_score: whole tumor dice score of teacher model 1
    param: t2_wt_score: whole tumor dice score of teacher model 2
    param: t3_wt_score: whole tumor dice score of teacher model 3
    
    return: weights: weights for teacher loss
    
    Description: calculate performance-based weights for teacher model losses
    """
    weights = [t1_wt_score, t2_wt_score, t3_wt_score]
    weights = [i / sum(weights) for i in weights]
    
    weights = [i * total_weight for i in weights]
    
    print("T1 weight: ", weights[0])
    print("T2 weight: ", weights[1])
    print("T3 weight: ", weights[2])
    
    return weights


def train_one_epoch(models, optimizers, loss_functions, lr_shedulars, train_loader, weights, epoch, device, writer):
    
    
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
    
    t1_wt_score = 0
    t2_wt_score = 0
    t3_wt_score = 0
    
    mean_loss = 0
    models['student_model'].train()
    models['teacher_model1'].train()
    models['teacher_model2'].train()
    models['teacher_model3'].train()
    
    for idx, (data, target) in enumerate(train_loader):
        data = data.to(device[0])
        target = target.to(device[0])
        
        output = models['student_model']((data )[:, 1, ...].unsqueeze(1))
        teacher_output1 = models["teacher_model1"](data[:, 0, ...].unsqueeze(1))
        teacher_output2 = models["teacher_model2"](data[:, 2, ...].unsqueeze(1))
        teacher_output3 = models["teacher_model3"](data[:, 3, ...].unsqueeze(1))
        
        
        ### teacher models loss calculation and backward pass
        teacher_loss1 = loss_functions['combination_loss'](target, teacher_output1.to(device[0]))
        teacher_loss2 = loss_functions['combination_loss'](target, teacher_output2.to(device[0]))
        teacher_loss3 = loss_functions['combination_loss'](target, teacher_output3.to(device[0]))
        
        ### calculating wt dice score for each teacher model
        temp_list = calculate_wt_dice_for_teacher_models(teacher_output1, teacher_output2, teacher_output3, target, device[0])
        t1_wt_score += temp_list[0]
        t2_wt_score += temp_list[1]
        t3_wt_score += temp_list[2]
        
        optimizers['teacher_optimizer1'].zero_grad()
        teacher_loss1.backward()
        optimizers['teacher_optimizer1'].step()
        
        optimizers['teacher_optimizer2'].zero_grad()
        teacher_loss2.backward()
        optimizers['teacher_optimizer2'].step()
        
        optimizers['teacher_optimizer3'].zero_grad()
        teacher_loss3.backward()
        optimizers['teacher_optimizer3'].step()
        
        
        
        
        ## KL divergence loss calculation and backward pass for student model
        if idx % 1 == 0 and epoch > 0:
            T = 20
            kl_divergence_loss_1 = kd_loss(output, teacher_output1.to(device[0]), T)
            kl_divergence_loss_2 = kd_loss(output, teacher_output2.to(device[0]), T)
            kl_divergence_loss_3 = kd_loss(output, teacher_output3.to(device[0]), T)
            KL_Loss = True
            
        else: 
            KL_Loss = False
            
        #loss = (dice_loss(target, output) + jaccard_loss(target, output) + ce_loss(output, target))/3.0
        loss = (loss_functions['combination_loss'](target, output) + weights[0] * kl_divergence_loss_1 + weights[1] *  
                kl_divergence_loss_2 + weights[2] * kl_divergence_loss_3) if KL_Loss else loss_functions['combination_loss'](target, output)
        
        
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
        
    t1_wt_score /= len(train_loader)
    t2_wt_score /= len(train_loader)
    t3_wt_score /= len(train_loader)
    
    weights = performance_base_weight_calculation(t1_wt_score, t2_wt_score, t3_wt_score)
    
    print(f"t1_wt_score: {t1_wt_score}")
    print(f"t2_wt_score: {t2_wt_score}")
    print(f"t3_wt_score: {t3_wt_score}")
    
    print("===========================================")
    return mean_loss / len(train_loader), weights


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
            data = data.to(device[0])
            target = target.to(device[0])
            
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
            
        
        if epoch >= 8:
            ## update learning rate
            print("Previous learning rate: ", lr_shedulars['plateau'].optimizer.param_groups[0]['lr'])
            lr_shedulars['plateau'].step(mean_loss / len(valid_loader))
            print("Learning rate: ", lr_shedulars['plateau'].optimizer.param_groups[0]['lr'])
        
        
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
    
    train_losses = []
    validitation_losses = []
    
    
    ### patience for early stopping
    patience = 20 #8
    weights = [0.1, 0.1, 0.1]
    
    for epoch in range(epochs):
        train_loss, weights = train_one_epoch(models, optimizers, loss_functions, lr_schedulars, train_loader,weights, epoch, device, writer)
        valid_loss, dice_dict = validitation_loss(models, loss_functions,lr_schedulars, valid_loader, epoch, device, writer,)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(models['student_model'].state_dict(),f"saved_models//{model_name}//best_loss_{fold}.pth")
        
        if dice_dict['mean'] > best_dice:
            best_dice = dice_dict['mean']
            torch.save(models['student_model'].state_dict(), f"saved_models//{model_name}//best_dice_{fold}.pth")
            
        
        ## save the model 
        torch.save(models['student_model'].state_dict(), f"saved_models//{model_name}//model_{fold}_{epoch}.pth")
        ## dump the dice dict to json file
        with open(f"results//{model_name}//dice_dict_{fold}_{epoch}.json", "w") as f:
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