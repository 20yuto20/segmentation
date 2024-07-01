import os
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import normalize
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import sys
import glob
import numbers
import random
import matplotlib.pyplot as plt

# import torchsummary
import tqdm
import time

# Add the parent dir to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from set_cfg import setup_config
from model.segnet import SegNet
from evalator import Evaluator
from dataloader import get_dataloader
from train_val import train, val, test
from utils.common import (
    setup_device,
    fixed_r_seed
)


def main(cfg):
    device = setup_device(cfg)
    fixed_r_seed(cfg)

    # TODO: managed by YAML
    batch_size = 5

    use_cuda = torch.cuda.is_available()
    print('Use CUDA:', use_cuda)

    # TODO: managed by YAML
    num_class = 41
    model = SegNet(input_channels=3, output_channels=num_class)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    #エポック数の設定
    # TODO: managed by YAML
    epoch_num = 400

    # 誤差関数の設定
    criterion = nn.CrossEntropyLoss(reduction='mean')
    criterion.to(device)

    train_loader, val_loader = get_dataloader()

    #評価関数
    evaluator = Evaluator(num_class)

    # 学習の実行
    loss_history = []
    start_time = time.time()

    for epoch in range(1, epoch_num+1):
        epoch_start_time = time.time()
        count = 0

        evaluator.reset()
        # tqdmを使用して学習の進捗を表示
        train_progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{epoch_num} [Train]')
        sum_loss = train()
        val_progress_bar = tqdm.tqdm(val_loader, desc=f'Epoch {epoch}/{epoch_num} [Val]')
        mIoU, Acc = val()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_duration = epoch_end_time - start_time
        
        print(f"Epoch: {epoch}, Loss: {sum_loss/(len(train_loader)*batch_size):.4f}, Accuracy: {Acc:.4f}, mIoU: {mIoU:.4f}")
        print(f"Epoch duration: {epoch_duration:.2f} seconds, Total duration: {total_duration:.2f} seconds")
        print("-" * 80)



    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")


if __name__ == "__main__":
    cfg = setup_config()
    main(cfg)