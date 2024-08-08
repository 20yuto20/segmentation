import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import numpy as np

def accuracy(output, target):
    pred = (output > 0.5).float()
    correct = (pred == target).float().sum()
    return correct / (target.size(0) * target.size(1))

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    train_acc = 0
    n_samples = 0
    
    all_outputs = []
    all_targets = []
    
    for batch in tqdm(train_loader):
        data, target = batch['image'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * data.size(0)
        train_acc += accuracy(output, target) # * data.size(0)
        n_samples += data.size(0)
        
        all_outputs.append(output.detach().cpu().numpy())
        all_targets.append(target.cpu().numpy())
        
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    ap_scores = []
    for i in range(all_targets.shape[1]):
        ap = average_precision_score(all_targets[:, i], all_outputs[:, i])
        ap_scores.append(ap)
    
    mAP = np.mean(ap_scores)
    
    return train_loss / n_samples, train_acc / n_samples, mAP

def val(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    val_acc = 0
    n_samples = 0
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            data, target = batch['image'].to(device), batch['label'].to(device)
            output = model(data)
            loss = criterion(output, target)
            
            val_loss += loss.item() * data.size(0)
            val_acc += accuracy(output, target) # * data.size(0)
            n_samples += data.size(0)
            
            all_outputs.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    ap_scores = []
    for i in range(all_targets.shape[1]):
        ap = average_precision_score(all_targets[:, i], all_outputs[:, i])
        ap_scores.append(ap)
    
    mAP = np.mean(ap_scores)
    
    return val_loss / n_samples, val_acc / n_samples, mAP

def test(model, device, test_loader, criterion, cfg):
    model.eval()
    test_loss = 0
    test_acc = 0
    n_samples = 0
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            data, target = batch['image'].to(device), batch['label'].to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item() * data.size(0)
            test_acc += accuracy(output, target) * data.size(0)
            n_samples += data.size(0)
            
            all_outputs.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    ap_scores = []
    for i in range(all_targets.shape[1]):
        ap = average_precision_score(all_targets[:, i], all_outputs[:, i])
        ap_scores.append(ap)
    
    mAP = np.mean(ap_scores)
    
    return test_loss / n_samples, test_acc / n_samples, mAP