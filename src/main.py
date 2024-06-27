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

from model.segnet import SegNet
from evalator import Evaluator
from dataloader import get_dataloader


batch_size = 5

use_cuda = torch.cuda.is_available()
print('Use CUDA:', use_cuda)

num_class = 41
model = SegNet(input_channels=3, output_channels=num_class)
if use_cuda:
    model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

#エポック数の設定
epoch_num = 400

# 誤差関数の設定
criterion = nn.CrossEntropyLoss(reduction='mean')
if use_cuda:
    criterion.cuda()

train_loader, val_loader = get_dataloader()

#評価関数
evaluator = Evaluator(num_class)

# 学習の実行
loss_history = []
start_time = time.time()

for epoch in range(1, epoch_num+1):
    epoch_start_time = time.time()
    sum_loss = 0.0
    count = 0
    evaluator.reset()
    # ネットワークを学習モードへ変更
    model.train()

    # tqdmを使用して学習の進捗を表示
    train_progress_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{epoch_num} [Train]')
    for sample in train_progress_bar:
        image, label = sample['image'], sample['label']
        if use_cuda:
            image = image.cuda()
            label = label.cuda()
        y = model(image)
        loss = criterion(y, label.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        train_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    # ネットワークを評価モードへ変更
    model.eval()
    # 評価の実行
    val_progress_bar = tqdm.tqdm(val_loader, desc=f'Epoch {epoch}/{epoch_num} [Val]')
    for sample in val_progress_bar:
        image, label = sample['image'], sample['label']
        if use_cuda:
            image = image.cuda()
            label = label.cuda()
        with torch.no_grad():
            y = model(image)

        loss = criterion(y, label.long())
        sum_loss += loss.item()
        pred = torch.argmax(y, dim=1)
        pred = pred.data.cpu().numpy()
        label = label.cpu().numpy()
        evaluator.add_batch(label, pred)
        val_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    mIoU = evaluator.Mean_Intersection_over_Union()
    Acc = evaluator.Pixel_Accuracy()
    
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    total_duration = epoch_end_time - start_time
    
    print(f"Epoch: {epoch}, Loss: {sum_loss/(len(train_loader)*batch_size):.4f}, Accuracy: {Acc:.4f}, mIoU: {mIoU:.4f}")
    print(f"Epoch duration: {epoch_duration:.2f} seconds, Total duration: {total_duration:.2f} seconds")
    print("-" * 80)

end_time = time.time()
total_training_time = end_time - start_time
print(f"Total training time: {total_training_time:.2f} seconds")