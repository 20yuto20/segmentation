import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
from utils.common import AverageMeter, intersectionAndUnionGPU
import time

def get_pred(y):
    if isinstance(y, torch.Tensor):
        out = y
    else:
        out = y[0]
    return torch.argmax(out, dim=1)

def denormalize_image(tensor: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """
    データセットで正規化した画像を可視化向けに逆正規化する関数.
    """
    for c in range(tensor.shape[0]):
        tensor[c] = tensor[c] * std[c] + mean[c]
    # [0,1]にクリップ
    tensor.clamp_(0.0, 1.0)
    return tensor

def visualize_results(cfg, epoch, image, label, pred, phase):
    debug_dir = os.path.join(cfg.out_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # バッチ内から最大3枚だけ可視化
    for j in range(min(3, image.shape[0])):  
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # ---- 修正ここから: 逆正規化してから表示 ----
        img_clone = image[j].clone().cpu()
        img_denorm = denormalize_image(img_clone, cfg.dataset.mean, cfg.dataset.std)
        img_np = img_denorm.permute(1, 2, 0).numpy()
        ax1.imshow(img_np)
        ax1.set_title("Input Image (denormalized)")
        # ---- 修正ここまで ----

        lbl = label[j]  # labelは既にNumPy配列
        ax2.imshow(lbl)
        ax2.set_title("True Label")
        
        prd = pred[j]   # predもNumPy配列
        ax3.imshow(prd)
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
            pred = output.argmax(1)
        elif output.dim() == 3:  # [B, H*W, C]
            pred = output.argmax(2).view(label.shape)
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")
        
        pred = pred.cpu().numpy()
        label = label.cpu().numpy()
        
        evaluator.add_batch(pred, label)
        
        loss_meter.update(loss.item(), image.size(0))
        train_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 25epoch毎に先頭バッチだけ可視化 (例)
        if epoch % 25 == 0 and i == 0:
            vis_image = image.cpu()      # (B, C, H, W)
            vis_label = label           # (B, H, W)  numpy
            vis_pred = pred            # (B, H, W)  numpy
            visualize_results(cfg, epoch, vis_image, vis_label, vis_pred, 'train')

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
            
            loss = criterion(output, label)
            loss_meter.update(loss.item(), image.size(0))
            
            pred = output.argmax(1)
            
            pred = pred.cpu().numpy()
            label = label.cpu().numpy()
            
            evaluator.add_batch(pred, label)
            
            val_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 同様に25epoch毎に先頭バッチだけ可視化 (例)
            if epoch % 25 == 0 and i == 0:
                vis_image = image.cpu()
                vis_label = label
                vis_pred = pred
                visualize_results(cfg, epoch, vis_image, vis_label, vis_pred, 'val')
                
    mIoU = evaluator.Mean_Intersection_over_Union()
    Acc = evaluator.Pixel_Accuracy()

    return loss_meter.avg, mIoU, Acc

def test(cfg, device, model, test_loader, criterion):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    test_progress_bar = tqdm.tqdm(test_loader, desc='Testing')

    total_inference_time = 0.0
    total_samples = 0
    
    for sample in test_progress_bar:
        image, label = sample['image'].to(device), sample['label'].to(device)

        if label.dim() == 4:
            label = label.squeeze(1)
        
        label = label.long()

        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(image)
        end_time = time.perf_counter()

        inference_time = end_time - start_time
        total_inference_time += inference_time
        total_samples += image.size(0)
        
        loss = criterion(output, label)
        pred = output.argmax(1)
        
        intersection, union, target = intersectionAndUnionGPU(pred, label, cfg.dataset.n_class, cfg.dataset.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        
        test_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    
    print(f"Test Results - Accuracy: {allAcc:.4f}, mIoU: {mIoU:.4f}")
    average_inference_time = total_inference_time / total_samples if total_samples > 0 else 0
    print(f"Test Inference Time(avg seconds / sample): {average_inference_time:.6f}")
    return mIoU, allAcc, average_inference_time
