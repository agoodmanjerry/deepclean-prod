
import numpy as np

import torch

from ..logger import Logger


def print_train():
    ''' Most important function '''
    print('------------------------------------')
    print('------------------------------------')
    print('')
    print('         ..oo0  ...ooOO00           ')
    print('        ..     ...             !!!  ')
    print('       ..     ...      o       \o/  ')
    print('   Y  ..     /III\    /L ---    n   ')
    print('  ||__II_____|\_/| ___/_\__ ___/_\__')
    print('  [[____\_/__|/_\|-|______|-|______|')
    print(' //0 ()() ()() 0   00    00 00    00')
    print('')
    print('------------------------------------')
    print('------------------------------------')
    

def get_device():
    ''' Convenient function to set up hardward '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory
        total_memory *= 1e-9 # convert bytes to Gb
        print('- Use device: {}'.format(torch.cuda.get_device_name(device)))
        print('- Total memory: {:.4f} GB'.format(total_memory))
    else:
        print('- Use device: CPU')
    return device


def train(train_loader, model, criterion, optimizer, lr_scheduler, 
          val_loader=None, max_epochs=1, logger=None, device='cpu'):
    """ Train network """
    
    # if Logger is not given, create a default logger
    if logger is None:
        logger = Logger(outdir='outdir', label='run', metrics=['loss'])
    
    # start training
    num_batches = len(train_loader)
    print_train()
    for epoch in range(max_epochs):
        
        # Training
        train_loss = 0.
        model.train()
        for i_batch, (data, target) in enumerate(train_loader):
            optimizer.zero_grad() # reset gradient
            
            # move batch to GPU if available
            data = data.to(device)
            target = target.to(device)
            
            # forward pass
            pred = model(data)
            
            # calculate loss function and backward pass
            loss = criterion(pred, target)
            loss.backward()
            
            # gradient descent
            optimizer.step()
            
            # Update training loss
            if criterion.reduction == 'mean':
                train_loss += loss.item() * len(data)
            else:
                train_loss += loss.item()
        
        # Evaluating performance on validation set if given
        val_loss = 0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for i_batch, (data, target) in enumerate(val_loader):
                    # move batch to GPU if available
                    data = data.to(device)
                    target = target.to(device)
                
                    # forward pass
                    pred = model(data)
                
                    # calculate and update validation loss
                    loss = criterion(pred, target)
                    if criterion.reduction == 'mean':
                        val_loss += loss.item() * len(data)
                    else:
                        val_loss += loss.item()
        
        # Compute average loss over all samples
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
            
        # Update LR with scheduler at the end of each epoch
        if lr_scheduler is not None:
            lr_scheduler.step()
       
        # Update metric, display, and save
        logger.update_metric(train_loss, val_loss, 'loss', epoch, 
                             num_batches, num_batches)
        logger.display_status(epoch, max_epochs, num_batches, num_batches,
                              train_loss, val_loss, 'loss')
        logger.log_metric()
        logger.save_model(model, epoch)
    
    
def evaluate(dataloader, model, criterion=None, device='cpu'):
    """ Inferring from dataloader. If `criterion` is given, also evaluate performance 
    of the network provided that `dataloader` includes target """
    model.eval()
    eval_loss = 0.
    prediction = []
    with torch.no_grad():
        for i_batch, (data, target) in enumerate(dataloader):
            # move to GPU if available
            data = data.to(device)
            target = target.to(device)
            
            # compute prediction
            pred = model(data)
            prediction.append(pred.cpu().numpy())
            
            # compute loss if criterion is given
            if criterion is not None:
                loss = criterion(pred, target)
                if criterion.reduction == 'mean':
                    eval_loss += loss.item() * len(data)  
                else:
                    eval_loss += loss.item()
    prediction = np.concatenate(prediction)
    
    # Averaging loss over the number of samples
    eval_loss /= len(dataloader.dataset)
    
    # Return prediction and loss (if criterion is given)
    if criterion is not None:
        return prediction, eval_loss
    return prediction
