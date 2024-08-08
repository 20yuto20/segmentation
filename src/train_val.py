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
        ax1.imshow(image[j].cpu().permute(1, 2, 0))
        ax1.set_title("Input Image")
        ax2.imshow(label[j].cpu())
        ax2.set_title("True Label")
        ax3.imshow(pred[j].cpu())
        ax3.set_title("Prediction")
        plt.savefig(os.path.join(debug_dir, f"debug_sample_epoch{epoch}_{phase}_sample{j}.png"))
        plt.close()

def train(cfg, device, model, train_progress_bar, optimizer, criterion, evaluator, epoch):
    model.train()
    evaluator.reset()
    loss_meter = AverageMeter()

    for i, sample in enumerate(train_progress_bar):
        image, label = sample['image'].to(device), sample['label'].to(device)
        if label.dim() == 4:
            label = label.squeeze(1)
        
        label = label.long()  # ラベルをLong型に変換
        
        output, main_loss, aux_loss = model(image, label)
        loss = main_loss + cfg.optimizer.loss.aux_weight * aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # メトリクスの計算
        if output.dim() == 4:  # [B, C, H, W]
            pred = output.argmax(1)  # クラスごとの最大値のインデックスを取得
        elif output.dim() == 3:  # [B, H*W, C]
            pred = output.argmax(2).view(label.shape)
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")
        
        # print(f"Pred shape: {pred.shape}, Label shape: {label.shape}")
        
        pred = pred.cpu().numpy()  # GPU tensor から numpy array に変換
        label = label.cpu().numpy()  # GPU tensor から numpy array に変換
        
        # print(f"Final pred shape: {pred.shape}, Final label shape: {label.shape}")
        
        evaluator.add_batch(pred, label)
        
        loss_meter.update(loss.item(), image.size(0))
        train_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        if epoch % 25 == 0 and i == 0:
            visualize_results(cfg, epoch, image, label, pred, 'train')

    mIoU = evaluator.Mean_Intersection_over_Union()
    Acc = evaluator.Pixel_Accuracy()

    return loss_meter.avg, mIoU, Acc

def val(cfg, device, model, val_progress_bar, criterion, evaluator, epoch):
    model.eval()
    evaluator.reset()
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for i, sample in enumerate(val_progress_bar):
            image, label = sample['image'].to(device), sample['label'].to(device)
            if label.dim() == 4:
                label = label.squeeze(1)
            
            label = label.long()
            output = model(image)
            
            # print(f"Output shape: {output.shape}, Label shape: {label.shape}")
            
            loss = criterion(output, label)
            loss_meter.update(loss.item(), image.size(0))
            
            pred = output.argmax(1)
            
            # print(f"Pred shape: {pred.shape}, Label shape: {label.shape}")
            
            pred = pred.cpu().numpy()
            label = label.cpu().numpy()
            
            evaluator.add_batch(pred, label)
            
            val_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            if epoch % 25 == 0 and i == 0:
                visualize_results(cfg, epoch, image, label, pred, 'val')

    mIoU = evaluator.Mean_Intersection_over_Union()
    Acc = evaluator.Pixel_Accuracy()

    return loss_meter.avg, mIoU, Acc

def test(cfg, device, model, test_loader, criterion):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    test_progress_bar = tqdm.tqdm(test_loader, desc='Testing')
    
    for sample in test_progress_bar:
        image, label = sample['image'].to(device), sample['label'].to(device)

        if label.dim() == 4:
            label = label.squeeze(1)
        
        label = label.long()

        with torch.no_grad():
            output = model(image)
        
        loss = criterion(output, label)
        pred = get_pred(output)
        
        intersection, union, target = intersectionAndUnionGPU(pred, label, cfg.dataset.n_class, cfg.dataset.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        
        test_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    
    print(f"Test Results - Accuracy: {allAcc:.4f}, mIoU: {mIoU:.4f}")
    return mIoU, allAcc