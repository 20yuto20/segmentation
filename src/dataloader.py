import numpy as np
import os
import torch
import random
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from augment import Cutout, Normalize, ToTensor, Normalize_Tensor, RandomCrop
from ra import RandAugmentSegmentation
from load_dataset.city import MYDataset
from load_dataset.voc import VOCDataset, make_datapath_list

# cfg.default.dataset_dir :  (SGE_LOCAL_DIR) + dataset/


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

    if cfg.dataset.name == "voc":
        # path_2012 = "/homes/ypark/code/dataset/VOCdevkit/VOC2012/"
        # path_2007 = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc/VOCdevkit/VOC2007"

        # abciで回すために少し変えました．
        path_2012 = cfg.default.dataset_dir + "VOCdevkit/VOC2012/"
        path_2007 = cfg.default.dataset_dir + "VOCdevkit/VOC2007/"

        print(f"load data from : {path_2007}")

        train_img_list, train_anno_list, val_img_list, val_anno_list, test_img_list, test_anno_list = make_datapath_list(
            path_2012=path_2012,
            path_2007=path_2007
            )
        
        train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=train_transform, img_size=cfg.dataset.resized_size)
        val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=val_transform, img_size=cfg.dataset.resized_size)
        ##### 追記をお願いします ############
        test_dataset = VOCDataset(test_img_list, test_anno_list, phase="test", transform=test_transform, img_size=cfg.dataset.resized_size)

    print(f"train dataset len : {len(train_dataset)}")
    print(f"val dataset len : {len(val_dataset)}")
    print(f"test dataset len : {len(test_dataset)}")

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

    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.learn.batch_size, 
        num_workers=cfg.default.num_workers, 
        shuffle=False,
        pin_memory=True
    )

    visualize_augmentations(cfg, train_dataset)

    return train_loader, val_loader, test_loader

def get_composed_transform(cfg, phase):
    transform_list = []

    if phase == "train":
        for aug_name in cfg.augment.name:
            if aug_name == "rcrop":
                transform_list.append(
                    RandomCrop(size=cfg.dataset.resized_size, padding=cfg.augment.hp.rcrop_pad)
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

def visualize_augmentations(cfg, train_dataset):
    """This function is for debugging and shows the differences between original one and augmented one by visualizing.

    Args:
        cfg (_type_): _description_
        train_dataset (_type_): _description_
    """
    output_dir = os.path.join(cfg.out_dir, "aug_samples")
    os.makedirs(output_dir, exist_ok=True)

    sample_indices = random.sample(range(len(train_dataset)), 5) # 一応5にしてあります。適宜欲しいサンプル数の数だけ値を変更してください。
    
    for idx in sample_indices:
        original_image, original_label = train_dataset.pull_item(idx)
        
        aug_sample = train_dataset[idx]
        aug_image = aug_sample['image']
        aug_label = aug_sample['label']

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        axs[0, 0].imshow(original_image)
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis('off')

        axs[0, 1].imshow(original_label, cmap='jet')
        axs[0, 1].set_title("Original Label")
        axs[0, 1].axis('off')

        axs[1, 0].imshow(aug_image.permute(1, 2, 0))
        axs[1, 0].set_title("Augmented Image")
        axs[1, 0].axis('off')
        
        axs[1, 1].imshow(aug_label, cmap='jet')
        axs[1, 1].set_title("Augmented Label")
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"augmentation_sample_{idx}.png"))
        plt.close()
        
    print(f"Augmentation samples saved to {output_dir}")