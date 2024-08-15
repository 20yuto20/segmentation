import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
from utils.common import AverageMeter, intersectionAndUnionGPU

def get_pred(y):
    if isinstance(y, torch.Tensor):
        out = y
    else:
        out = y[0]
    return torch.argmax(out, dim=1)

def visualize_results(cfg, epoch, image, label, pred, phase):
    debug_dir = os.path.join(cfg.out_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    for j in range(min(3, image.shape[0])):  # Visualize up to 3 samples
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Convert image to numpy if it's a tensor
        if isinstance(image, torch.Tensor):
            img = image[j].cpu().permute(1, 2, 0).numpy()
        else:
            img = image[j].transpose(1, 2, 0)
        
        ax1.imshow(img)
        ax1.set_title("Input Image")
        
        # Convert label to numpy if it's a tensor
        if isinstance(label, torch.Tensor):
            lbl = label[j].cpu().numpy()
        else:
            lbl = label[j]
        
        ax2.imshow(lbl)
        ax2.set_title("True Label")
        
        # Convert pred to numpy if it's a tensor
        if isinstance(pred, torch.Tensor):
            prd = pred[j].cpu().numpy()
        else:
            prd = pred[j]
        
        ax3.imshow(prd)
        ax3.set_title("Prediction")
        
        plt.savefig(os.path.join(debug_dir, f"debug_sample_epoch{epoch}_{phase}_sample{j}.png"))
        plt.close()

def train(cfg, device, model, train_loader, optimizer, criterion, evaluator, epoch):
    model.train()
    evaluator.reset()
    loss_meter = AverageMeter()
    
    train_progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{cfg.learn.n_epoch} [Train]')

    for i, sample in enumerate(train_loader):
        image, label = sample['image'].to(device), sample['label'].to(device)
        if label.dim() == 4:
            label = label.squeeze(1)
        
        label = label.long()
        
        output, main_loss, aux_loss = model(image, label)
        loss = main_loss + cfg.optimizer.loss.aux_weight * aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # メトリクスの計算
        pred = output.argmax(1)
        evaluator.add_batch(pred.cpu().numpy(), label.cpu().numpy())
        
        loss_meter.update(loss.item(), image.size(0))
        train_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        if epoch % 25 == 0 and i == 0:
            visualize_results(cfg, epoch, image, label, pred, 'train')

    mIoU = evaluator.Mean_Intersection_over_Union()
    Acc = evaluator.Pixel_Accuracy()

    return loss_meter.avg, mIoU, Acc

def val(cfg, device, model, val_loader, criterion, evaluator, epoch):
    model.eval()
    evaluator.reset()
    loss_meter = AverageMeter()
    
    val_progress_bar = tqdm.tqdm(val_loader, desc=f'Epoch {epoch}/{cfg.learn.n_epoch} [Val]')
    
    with torch.no_grad():
        for i, sample in enumerate(val_progress_bar):
            image, label = sample['image'].to(device), sample['label'].to(device)
            if label.dim() == 4:
                label = label.squeeze(1)
            
            label = label.long()
            output = model(image)
            
            loss = criterion(output, label)
            loss_meter.update(loss.item(), image.size(0))
            
            pred = get_pred(output)
            
            evaluator.add_batch(pred.cpu().numpy(), label.cpu().numpy())
            
            val_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            if epoch % 25 == 0 and i == 0:
                visualize_results(cfg, epoch, image, label, pred, 'val')
                
    mIoU = evaluator.Mean_Intersection_over_Union()
    Acc = evaluator.Pixel_Accuracy()

    return loss_meter.avg, mIoU, Acc

def test(cfg, device, model, test_loader, criterion):
    model.eval()
    evaluator = Evaluator(cfg.dataset.n_class)
    loss_meter = AverageMeter()
    
    test_progress_bar = tqdm.tqdm(test_loader, desc='Testing')
    
    with torch.no_grad():
        for sample in test_progress_bar:
            image, label = sample['image'].to(device), sample['label'].to(device)
            if label.dim() == 4:
                label = label.squeeze(1)
            
            label = label.long()
            output = model(image)
            
            loss = criterion(output, label)
            loss_meter.update(loss.item(), image.size(0))
            
            pred = get_pred(output)
            
            evaluator.add_batch(pred.cpu().numpy(), label.cpu().numpy())
            
            test_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    mIoU = evaluator.Mean_Intersection_over_Union()
    Acc = evaluator.Pixel_Accuracy()
    
    print(f"Test Results - Loss: {loss_meter.avg:.4f}, Accuracy: {Acc:.4f}, mIoU: {mIoU:.4f}")
    return mIoU, Acc