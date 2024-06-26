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
import torchsummary
import cv2
import tqdm
import time

print("hello")
print("sub")

use_cuda = torch.cuda.is_available()
print('Use CUDA:', use_cuda)

num_class = 41
model = SegNet(input_channels=3, output_channels=num_class)
if use_cuda:
    model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

#エポック数の設定
epoch_num = 10

# 誤差関数の設定
criterion = nn.CrossEntropyLoss(reduction='mean')
if use_cuda:
    criterion.cuda()

#モデルの情報を表示
torchsummary.summary(model,(3,128,128))

# 学習済みモデルを呼び出す

# load_path = "./ARCdataset_png/checkpoint.pth.tar"
# checkpoint = torch.load(load_path)
# model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])

#評価関数
evaluator = Evaluator(num_class)
# 学習の実行
loss_history=[]
for epoch in range(1, epoch_num+1):
    sum_loss = 0.0
    count = 0
    evaluator.reset()
    # ネットワークを学習モードへ変更
    model.train()

    for sample in train_loader:

        image, label = sample['image'], sample['label']
        if use_cuda:
            image = image.cuda()
            label = label.cuda()
        y = model(image)
        loss = criterion(y, label.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #sum_loss += loss.item()

    # ネットワークを評価モードへ変更
    model.eval()
    # 評価の実行
    for sample in val_loader:
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
       
    #img_size = image.size()
    #loss_history.append(sum_loss)
    mIoU = evaluator.Mean_Intersection_over_Union()
    Acc = evaluator.Pixel_Accuracy()
    print("epoch: {}, mean loss: {}, mean accuracy: {}，　mean IoU: {}".format(epoch, sum_loss/(len(train_loader)*batch_size), Acc, mIoU))

