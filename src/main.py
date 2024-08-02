import os
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
import tqdm
import time
import pandas as pd

from set_cfg import setup_config, add_config
from evalator import Evaluator
from dataloader import get_dataloader
from train_val import train, val
from utils.common import (
    setup_device,
    fixed_r_seed,
    get_time,
    plot_log,
    save_learner  
)
from utils.suggest import (
    suggest_network,
    suggest_loss_func,
    suggest_optimizer,
    suggest_scheduler
)

# Add the parent dir to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def visualize_samples(dataloader, num_samples=5):
    samples = next(iter(dataloader))
    images, labels = samples['image'], samples['label']

    # color_palette = []
    # for i in range(20):
    #     color_palette.append([i, i, i])
    # color_palette = np.array(color_palette)

    for i in range(min(num_samples, len(images))):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # 画像の表示
        img = images[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())  # 正規化
        ax1.imshow(img)
        ax1.set_title("Input Image")
        
        # ラベルの表示
        label = labels[i].squeeze().numpy()  # チャンネル次元を削除
        # label = color_palette[label]    # カラーパレットに従ってRGBに変更
        # label = (label - label.min()) / (label.max() - label.min())
        # ax2.imshow(label, cmap='jet')  # カラーマップを使用
        ax2.imshow(label)
        ax2.set_title("Label")
        
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(cur_dir)

        save_dir = os.path.join(parent_dir, "output", "sample")
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, f"sample_visualization_{i}.png")
        
        plt.savefig(file_path)
        plt.close()


def main(cfg):
    device = setup_device(cfg)
    fixed_r_seed(cfg)

    model = suggest_network(cfg)
    model.to(device)

    optimizer = suggest_optimizer(cfg, model)
    scheduler = suggest_scheduler(cfg, optimizer)

    criterion = suggest_loss_func(cfg)
    criterion.to(device)

    train_loader, val_loader, _ = get_dataloader(cfg)

    evaluator = Evaluator(cfg.dataset.n_class)

    all_training_result = []
    start_time = time.time()
    best_miou = 0.0

    for epoch in range(1, cfg.learn.n_epoch+1):
        train_progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{cfg.learn.n_epoch} [Train]')
        train_loss, train_mIoU, train_Acc = train(cfg, device, model, train_progress_bar, optimizer, criterion, evaluator, epoch)
    
        val_progress_bar = tqdm.tqdm(val_loader, desc=f'Epoch {epoch}/{cfg.learn.n_epoch} [Val]')
        val_mIoU, val_Acc = val(cfg, device, model, val_progress_bar, criterion, evaluator, epoch)

        all_training_result.append({
            "epoch": epoch, 
            "train_loss": train_loss, 
            "train_mIoU": train_mIoU, 
            "train_acc": train_Acc,
            "val_mIoU": val_mIoU, 
            "val_acc": val_Acc
        })

        epoch_end_time = time.time()
        total_duration = get_time(epoch_end_time - start_time)

        print(f"{total_duration}, lr : {optimizer.param_groups[0]['lr']}")
        print(f"Epoch: {epoch}, Loss: {train_loss:.4f}, Train Accuracy: {train_Acc:.4f}, Train mIoU: {train_mIoU:.4f}")
        print(f"Val Accuracy: {val_Acc:.4f}, Val mIoU: {val_mIoU:.4f}")
        print("-" * 80)

        if val_mIoU > best_miou:
            best_miou = val_mIoU
            print(f"New best mIoU: {best_miou}. Saving model...")
            save_learner(cfg, model, device, True)
            
        scheduler.step()

    end_time = time.time()
    total_training_time = get_time(end_time - start_time)
    print(f"Total training {total_training_time}")

    if len(all_training_result) > 0:
        train_df = pd.DataFrame(all_training_result)
        train_df.to_csv(cfg.out_dir + "train_output.csv", index=False)
        plot_log(cfg, train_df)

    add_config(cfg, {"total_training_time": str(total_training_time['time'])})

if __name__ == "__main__":
    cfg = setup_config()
    main(cfg)