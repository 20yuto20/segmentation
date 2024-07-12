import numpy as np
import os
import torch
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from augment import Cutout, Normalize, ToTensor, Normalize_Tensor
from ra import RandAugmentSegmentation
from load_dataset.city import MYDataset
from load_dataset.voc import VOCDataset, make_datapath_list



def get_dataloader(cfg):
    cur_dir = Path(__file__).parent.parent
    dataset_name = cfg.dataset.name
    dataset_path = str(cur_dir / "dataset" / dataset_name)
    
    train_transform = get_composed_transform(cfg, "train")
    val_transform = get_composed_transform(cfg, "val")
    test_transform = get_composed_transform(cfg, "test")

    # train_dataset = MYDataset(dataset_path, split='train', transform=train_transform)
    # val_dataset = MYDataset(dataset_path, split='val', transform=val_transform)
    # test_dataset = MYDataset(dataset_path, split='test', transform=test_transform)

    # データセット作成

    # とりあえずif文で書きました，あとで適宜修正
    if cfg.dataset.name == "voc":
        rootpath = "/homes/ypark/code/dataset/VOCdevkit/VOC2012/"
        train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
            rootpath=rootpath)
        
        train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=train_transform, img_size=cfg.dataset.resized_size)
        val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=val_transform, img_size=cfg.dataset.resized_size)
        ##### 追記をお願いします ############
        # test_dataset = 


    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.learn.batch_size, 
        num_workers=cfg.default.num_workers, 
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.learn.batch_size, 
        num_workers=cfg.default.num_workers, 
        shuffle=False,
        pin_memory=True
    )

    # test_loader = DataLoader(
    #     test_dataset, 
    #     batch_size=cfg.learn.batch_size, 
    #     num_workers=cfg.default.num_workers, 
    #     shuffle=False,
    #     pin_memory=True
    # )

    ### とりあえず，testもvalを返す

    # return train_loader, val_loader, test_loader
    return train_loader, val_loader, val_loader

def get_composed_transform(cfg, phase):
    transform_list = []

    if phase == "train":
        for aug_name in cfg.augment.name:
            if aug_name == "rcrop":
                transform_list.append(
                    lambda x: {'image': transforms.RandomCrop(size=cfg.dataset.resized_size, padding=cfg.augment.hp.rcrop_pad)(x['image']),
                               'label': transforms.RandomCrop(size=cfg.dataset.resized_size, padding=cfg.augment.hp.rcrop_pad)(x['label'])}
                )
            elif aug_name == "hflip":
                transform_list.append(
                    lambda x: {'image': transforms.RandomHorizontalFlip(p=0.5)(x['image']),
                               'label': transforms.RandomHorizontalFlip(p=0.5)(x['label'])}
                )
            elif aug_name == "cutout":
                transform_list.append(
                    transforms.RandomApply(
                        [lambda x: {'image': Cutout(n_holes=1, img_size=cfg.dataset.resized_size, patch_size=cfg.augment.hp.cutout_size)(x['image']),
                                    'label': x['label']}],
                        p=cfg.augment.hp.cutout_p
                    )
                )
            elif aug_name == "ra":
                transform_list.append(RandAugmentSegmentation(cfg=cfg, num_ops=cfg.augment.ra.num_op, magnitude=cfg.augment.ra.magnitude))
            elif aug_name == "nan":
                pass
            else:
                raise ValueError(f"Invalid Augment ... {aug_name}")
    
    if cfg.dataset.name == "voc":
        transform_list.append(Normalize_Tensor(color_mean=cfg.dataset.mean, color_std=cfg.dataset.std))
    else:
        transform_list.append(ToTensor())
        transform_list.append(Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std))

    transform_list = transforms.Compose(transform_list)

    return transform_list



