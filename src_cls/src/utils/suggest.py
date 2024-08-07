import random
import numpy as np
# from timm.scheduler.cosine_lr import CosineLRScheduler

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vgg16_bn

from models.cifar_resnet import ResNetBasicBlock, ResNetBottleneck
from models.trivial_wide import TrivialWideResNet
from models.resnet import resnet50


def setup_device(cfg):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.default.device_id}")
        
        if not cfg.default.deterministic:
            torch.backends.cudnn.benchmark = True

    else:
        device = "cpu"

    print("CUDA is available:", torch.cuda.is_available())
    print(f"using device: {device}")
    return device


def fixed_r_seed(cfg):
    random.seed(cfg.default.seed)
    np.random.seed(cfg.default.seed)
    torch.manual_seed(cfg.default.seed)
    torch.cuda.manual_seed(cfg.default.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def suggest_network(cfg):
    if cfg.network.name == "resnet50":
        model = resnet50(pretrained=cfg.network.pretrained)
        model.fc = nn.Linear(model.fc.in_features, cfg.dataset.n_class)

    elif cfg.network.name == "vit":
        model = vit_b_16(weights = "ViT_B_16_Weights.IMAGENET1K_V1")
        model.heads[0] = nn.Linear(model.heads[0].in_features, cfg.dataset.n_class)
  
    elif cfg.network.name == "vgg16":
        model = vgg16_bn()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, cfg.dataset.n_class)

    elif cfg.network.name == "resnet56":
        # 'When use Bottleneck, depth should be 9n+2 (e.g. 47, 56, 110, 1199).'
        model = ResNetBottleneck(depth=56, n_class=cfg.dataset.n_class)    # BasicBlock構造を用いる場合

    # 'When use basicblock, depth should be 6n+2 (e.g. 20, 32, 44).'
    elif cfg.network.name == "resnet20":
        model = ResNetBasicBlock(depth=20, n_class=cfg.dataset.n_class)   # Bottleneck構造を用いる場合

    elif cfg.network.name == 'wrn28_10':
        model = TrivialWideResNet(
            28, 10, 
            dropout_rate=cfg.network.dropout_rate, 
            num_classes=cfg.dataset.n_class, 
            adaptive_dropouter_creator=None,
            adaptive_conv_dropouter_creator=None, 
            groupnorm=False, 
            examplewise_bn=False, 
            virtual_bn=False
            )
        
    return model


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
        
    # elif cfg.optimizer.scheduler.name == "warmup":
    #     scheduler = CosineLRScheduler(
    #         optimizer,
    #         t_initial=cfg.learn.n_epoch,
    #         lr_min=cfg.optimizer.hp.lr_min, 
    #         warmup_t=cfg.optimizer.hp.warmup_period,
    #         warmup_lr_init=cfg.optimizer.hp.warmup_init,
    #         warmup_prefix=False
    #         )
    
    else:
        raise ValueError(f"Invalid Lr Scheduler ... {cfg.optimizer.scheduler.name}, select from < cosine, step >")
    
    return scheduler


def suggest_loss_func():
    return nn.BCEWithLogitsLoss(reduction='mean')












