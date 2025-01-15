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
from train_val import train, val, test
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

def denormalize_image(tensor: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """
    データローダー側で正規化したイメージテンソルを逆正規化して可視化向けに戻す.
    tensor: (C, H, W), 値は標準化済み
    mean, std: cfg.dataset.mean, cfg.dataset.std
    """
    # ここでは mean, std が RGB 各チャネルの値を想定
    for c in range(tensor.shape[0]):
        tensor[c] = tensor[c] * std[c] + mean[c]
    # 値を [0,1] 範囲に制限 (正確には元データの状況に合わせてクリップ)
    tensor.clamp_(0.0, 1.0)
    return tensor

def visualize_samples(dataloader, num_samples=5, mean=None, std=None):
    """サンプル画像を描画する関数。色味が変わらないように逆正規化して可視化する。"""
    samples = next(iter(dataloader))
    images, labels = samples['image'], samples['label']

    # num_samplesだけ可視化
    for i in range(min(num_samples, len(images))):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # ---- 修正ここから: min-maxスケーリングを廃止し、denormalize ----
        # (C, H, W) → (H, W, C) に変換してから可視化
        img_clone = images[i].clone()
        img_denorm = denormalize_image(img_clone, mean, std)  # 逆正規化
        img_np = img_denorm.permute(1, 2, 0).cpu().numpy()    # NumPy化
        ax1.imshow(img_np)
        ax1.set_title("Input Image (denormalized)")
        # ---- 修正ここまで ----

        # ラベル表示
        label = labels[i].squeeze().cpu().numpy()
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

    train_loader, val_loader, test_loader = get_dataloader(cfg)

    # optional: visualize a few samples from train_loader
    # ここで画像を可視化する際、逆正規化にcfgのmeanとstdを渡す
    # mean, stdを必ずlistかタプルで渡す
    visualize_samples(train_loader, num_samples=3, mean=cfg.dataset.mean, std=cfg.dataset.std)

    evaluator = Evaluator(cfg.dataset.n_class)

    all_training_result = []
    start_time = time.time()
    best_miou = 0.0

    for epoch in range(1, cfg.learn.n_epoch+1):
        train_progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{cfg.learn.n_epoch} [Train]')
        train_loss, train_mIoU, train_Acc = train(cfg, device, model, train_progress_bar, optimizer, criterion, evaluator, epoch)
    
        val_progress_bar = tqdm.tqdm(val_loader, desc=f'Epoch {epoch}/{cfg.learn.n_epoch} [Val]')
        val_loss, val_mIoU, val_Acc = val(cfg, device, model, val_progress_bar, criterion, evaluator, epoch)

        all_training_result.append({
            "epoch": epoch, 
            "train_loss": train_loss, 
            "train_mIoU": train_mIoU, 
            "train_acc": train_Acc,
            "val_loss": val_loss,
            "val_mIoU": val_mIoU, 
            "val_acc": val_Acc
        })

        epoch_end_time = time.time()
        total_duration = get_time(epoch_end_time - start_time)

        print(f"{total_duration}, lr : {optimizer.param_groups[0]['lr']}")
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_Acc:.4f}, Train mIoU: {train_mIoU:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_Acc:.4f}, Val mIoU: {val_mIoU:.4f}")
        print("-" * 80)

        if val_mIoU > best_miou:
            best_miou = val_mIoU
            print(f"New best mIoU: {best_miou}. Saving model...")
            save_learner(cfg, model, device, True)
            
        scheduler.step()

    end_time = time.time()
    total_training_time = get_time(end_time - start_time)
    print(f"Total training {total_training_time}")
    
    # best model を読み込み
    best_model_path = cfg.out_dir + "weights/best.pth"
    model.load_state_dict(torch.load(best_model_path))

    # テスト実行
    test_mIoU, test_Acc, average_inference_time = test(cfg, device, model, test_loader, criterion)
    print(f"Final Test Results - Test Accuracy: {test_Acc:.4f}, Test mIoU: {test_mIoU:.4f}")

    # 結果を保存
    test_result = {
        "test_mIoU": test_mIoU, 
        "test_Acc": test_Acc,
        "avg_inference_time": average_inference_time
    }

    if len(all_training_result) > 0:
        train_df = pd.DataFrame(all_training_result)
        train_df.to_csv(cfg.out_dir + "train_output.csv", index=False)
        plot_log(cfg, train_df)

    test_df = pd.DataFrame([test_result])
    test_df.to_csv(cfg.out_dir + "test_output.csv", index=False)

    print(f"Train results saved to: {cfg.out_dir}train_output.csv")
    print(f"Test results saved to: {cfg.out_dir}test_output.csv")

    add_config(cfg, {
        "test_acc": float(test_Acc), 
        "test_mIoU": float(test_mIoU),
        "avg_inference_time": float(average_inference_time)
    })
    add_config(cfg, {"total_training_time": str(total_training_time['time'])})

if __name__ == "__main__":
    cfg = setup_config()
    main(cfg)
