import torch




from metrics import calculate_dice_score, calculate_hd95_multi_class, save_history, multiclass_dice_coeff



def train_one_epoch(model, optimizer, dice_loss, jaccard_loss, ce_loss, kl_divergence, train_loader, epoch, device, writer):
    
    
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
    mean_loss = 0
    model.train()
    
    for idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        output = model(data)[0]
        
        loss = (dice_loss(target, output) + jaccard_loss(target, output) + ce_loss(output, target))/3.0
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
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
            print("===========================================")
        
    return mean_loss / len(train_loader)


def validitation_loss(model, dice_loss, jaccard_loss, ce_loss, kl_divergence, valid_loader, epoch, device, writer):
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
    model.eval()
    
    dice_dict = {}
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(valid_loader):
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)[0]
            
            loss = (dice_loss(target, output) + jaccard_loss(target, output) + ce_loss(output, target))/3.0
            
            mean_loss += loss.detach().cpu().item()
            
          
            preds = torch.softmax(output, dim=1)
            temp_dice_dict = multiclass_dice_coeff(preds=preds, target=target)
            dice_dict['mean'] += temp_dice_dict['mean']
            dice_dict['N-NE'] += temp_dice_dict['N-NE']
            dice_dict['ED'] += temp_dice_dict['ED']
            dice_dict['ET'] += temp_dice_dict['ET']
            
    
        dice_dict['mean'] /= len(valid_loader)
        dice_dict['N-NE'] /= len(valid_loader)
        dice_dict['ED'] /= len(valid_loader)
        dice_dict['ET'] /= len(valid_loader)
        
        
        print(f"Epoch: {epoch} | Valid Loss: {mean_loss / len(valid_loader)}")
        print("mean loss: ", mean_loss / len(valid_loader))
        print(f"dice mean score: {dice_dict['mean']}")
        print(f"N-NE dice score: {dice_dict['N-NE']}")
        print(f"ED dice score: {dice_dict['ED']}")
        print(f"ET dice score: {dice_dict['ET']}")
        
        print("===========================================")
        
        
        writer.add_scalar("Loss/valid", mean_loss / len(valid_loader), epoch)
        
    return mean_loss / len(valid_loader), dice_dict



def Fit(model, optimizer, dice_loss, jaccard_loss, ce_loss, kl_divergence, train_loader, valid_loader, epochs, device, writer):
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
    
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, optimizer, dice_loss, jaccard_loss, ce_loss, kl_divergence, train_loader, epoch, device, writer)
        valid_loss, dice_dict = validitation_loss(model, dice_loss, jaccard_loss, ce_loss, kl_divergence, valid_loader, epoch, device, writer)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), "best_loss.pth")
        
        if dice_dict['mean'] > best_dice:
            best_dice = dice_dict['mean']
            torch.save(model.state_dict(), "best_dice.pth")
            
        save_history(train_loss, valid_loss, dice_dict, epoch)
        
        train_losses.append(train_loss)
        validitation_losses.append(valid_loss)
        
        
    history = {'epochs': epochs, 'train_loss': train_losses, 'valid_loss': validitation_losses}
        
    writer.close()
    print("Training Finished")
    
    return history