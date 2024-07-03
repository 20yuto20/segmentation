import os
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import sys
import matplotlib.pyplot as plt
# import torchsummary
import torch
import tqdm
import time
import pandas as pd


from set_cfg import setup_config, add_config
from model.segnet import SegNet
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


# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# parent_dir : ['/homes/ypark/code/segmentation/src/']


def main(cfg):
    device = setup_device(cfg)
    fixed_r_seed(cfg)

    model = suggest_network(cfg)
    model.to(device)

    optimizer = suggest_optimizer(cfg, model)
    scheduler = suggest_scheduler(cfg, optimizer)

    # 誤差関数の設定
    criterion = suggest_loss_func()
    criterion.to(device)

    train_loader, val_loader, test_loader = get_dataloader(cfg)

    #評価関数
    evaluator = Evaluator(cfg.dataset.n_class)

    # 学習の実行
    all_training_result = []
    start_time = time.time()
    best_miou = 0.0

    for epoch in range(1, cfg.learn.n_epoch+1):
        evaluator.reset()
        # tqdmを使用して学習の進捗を表示
        train_progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{cfg.learn.n_epoch} [Train]')
        loss = train(device, model, train_progress_bar, optimizer, criterion)
    
        val_progress_bar = tqdm.tqdm(val_loader, desc=f'Epoch {epoch}/{cfg.learn.n_epoch} [Val]')
        mIoU, Acc = val(device, model, val_progress_bar, criterion, evaluator)

        all_training_result.append([loss, mIoU, Acc])

        epoch_end_time = time.time()
        total_duration = get_time(epoch_end_time - start_time) # dict型

        # print(f"Epoch: {epoch}, Loss: {sum_loss/(len(train_loader)*cfg.learn.batch_size):.4f}, Accuracy: {Acc:.4f}, mIoU: {mIoU:.4f}")
        print(f"{total_duration}, lr : {optimizer.param_groups[0]['lr']}")
        print(f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {Acc:.4f}, mIoU: {mIoU:.4f}")
        print("-" * 80)

        # 最良のval mIoUの時にモデル重みを保存
        if mIoU > best_miou:
            best_miou = mIoU
            print(f"New best mIoU: {best_miou}. Saving model...")
            save_learner(cfg, model, device, True)
            
        scheduler.step()


    end_time = time.time()
    total_training_time = get_time(end_time - start_time) # dict型
    print(f"Total training {total_training_time}")

    # Load the best model
    best_model_path = cfg.out_dir + "weights/best.pth"
    model.load_state_dict(torch.load(best_model_path))

    # Run test
    test_mIoU, test_Acc = test(cfg, device, model, test_loader, criterion, evaluator)
    print(f"Final Test Results - Test Accuracy: {test_Acc:.4f}, Test mIoU: {test_mIoU:.4f}")
    all_training_result.append([test_mIoU, test_Acc])

    # 学習結果をdfに，csvファイルに保存
    all_training_result = pd.DataFrame(
        np.array(all_training_result),
        columns=["train_loss", "val_mIoU", "val_acc", "test_mIoU", "test_Acc"],
    )
    save_file_path = cfg.out_dir + "output.csv"
    all_training_result.to_csv(save_file_path, index=False)

    # TODO: add_configの使い方がこれであっているか確認
    add_config(cfg, {"test_acc" : test_Acc})
    add_config(cfg, total_training_time)
    plot_log(cfg, all_training_result)



if __name__ == "__main__":
    cfg = setup_config()
    main(cfg)