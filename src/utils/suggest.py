import random
import numpy as np

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vgg16_bn

from model.segnet import SegNet
from model.pspnet import PSPNet
from loss.entropy import CrossEntropyLoss, PSPLoss
from utils.lr_scheduler import PolyLR # polynomialスケジューラーのアルゴリズムが記載されている



# configをもとに使用モデルを識別，呼び出す
# 後にsegnet以外も使いたい時があるかも
def suggest_network(cfg):
    if cfg.network.name == "segnet":
        print("Selected network is a SEGNET!")
        model = SegNet(input_channels=3, output_channels=cfg.dataset.n_class)
    elif cfg.network.name == "pspnet":
        print("Selected network is a PSPNET!")
        model = PSPNet(n_classes=cfg.dataset.n_class)

    return model


# configをもとに最適化手法を設定
# 学習率などのハイパラもconfigを参照
def suggest_optimizer(cfg, model):
    if cfg.optimizer.name == "SGD":
        optimizer = torch.optim.SGD(
            params=model.parameters(), 
            lr=cfg.optimizer.hp.lr, 
            momentum=cfg.optimizer.hp.momentum, 
            weight_decay=cfg.optimizer.hp.weight_decay, 
            nesterov=True,
        )
    elif cfg.optimizer.name == "AdamW":
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr = cfg.optimizer.hp.lr,
            weight_decay=cfg.optimizer.hp.weight_decay,
        )
    elif cfg.optimizer.name == "Adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr = cfg.optimizer.hp.lr,
            weight_decay=cfg.optimizer.hp.weight_decay,
        )
        
    else:
        raise ValueError(f"Invalid optimizer ... {cfg.optimizer.name}, Select from < SGE, Adam >.")
    
    return optimizer


# 学習率の設定，基本的にはcosine decayで良いのでは.. -> polyが最適！！
def suggest_scheduler(cfg, optimizer):
    if cfg.optimizer.scheduler.name == "fix":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[],
            gamma=1.0,
        )

    elif cfg.optimizer.scheduler.name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.learn.n_epoch,
            eta_min=cfg.optimizer.hp.lr_min,
        )

    elif cfg.optimizer.scheduler.name == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.optimizer.scheduler.step,
            gamma=0.1,
        )

    elif cfg.optimizer.scheduler.name == "poly":
        scheduler = PolyLR(
            optimizer,
            T_max=cfg.learn.n_epoch,
            eta_min=cfg.optimizer.hp.lr_min,
            power=cfg.optimizer.scheduler.power
        )
        
   
    else:
        raise ValueError(f"Invalid Lr Scheduler ... {cfg.optimizer.scheduler.name}, select from < cosine, step >")
    
    return scheduler


# loss/entropy.pyの中に定義されているCrossEntropyLossクラスを呼び出す
def suggest_loss_func(cfg):
    if cfg.optimizer.loss.name == "ce":
        if cfg.network.name == "pspnet":
            criterion = PSPLoss(cfg)
        else:
            criterion = CrossEntropyLoss(cfg)
    else: 
        raise ValueError("incorrect loss name ... available : ce")
    
    return criterion












