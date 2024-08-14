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
from load_dataset.voc import VOCDataset, datapath_list
# cfg.default.dataset_dir :  (SGE_LOCAL_DIR) + dataset/


def get_dataloader(cfg):
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
        # path_2012 = cfg.default.dataset_dir + "VOCSBD/"
        # path_2007 = cfg.default.dataset_dir + "VOC2007/VOCdevkit/VOC2007/"
        
        path_train = cfg.default.dataset_dir + "train_aug/"
        path_val = cfg.default.dataset_dir + "val/"
        # path_test = cfg.default.dataset_dir + "test/"
        path_test = cfg.default.dataset_dir + "test_2007/"


        print(f"load train from : {path_train} \n load validation from : {path_val} \n load test from : {path_test}")

        train_img_list, train_anno_list, val_img_list, val_anno_list, test_img_list, test_anno_list = datapath_list(
            path_train=path_train,
            path_val=path_val,
            path_test=path_test
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
            elif  aug_name == "hflip":
                transform_list.append(
                    transforms.RandomApply(
                        [lambda x: {
                            'image': ImageOps.mirror(x['image']),
                            'label': ImageOps.mirror(x['label'])
                        }],
                        p=0.5  # 50%の確率で適用
                    )
                )
            
            elif aug_name == "vflip":
                transform_list.append(
                    transforms.RandomApply(
                        [lambda x: {
                            'image': ImageOps.flip(x['image']),
                            'label': ImageOps.flip(x['label'])
                        }],
                        p=0.5  # 50%の確率で適用
                    )
                )
                # 50%
                # transform_list.append(
                #     lambda x: {
                #         'image': x['image'].flip(-1) if random.random() < 0.5 else x['image'],
                #         'label': x['label'].flip(-1) if x['image'].flip(-1).equal(x['image']) else x['label']
                #     }
                # )
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

def get_voc_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
    return colormap

def visualize_label(label, colormap):
    r = label.copy()
    g = label.copy()
    b = label.copy()
    for l in range(0, len(colormap)):
        r[label == l] = colormap[l, 0]
        g[label == l] = colormap[l, 1]
        b[label == l] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def visualize_augmentations(cfg, train_dataset):
    output_dir = os.path.join(cfg.out_dir, "aug_samples")
    os.makedirs(output_dir, exist_ok=True)

    sample_indices = random.sample(range(len(train_dataset)), 5)
    
    def denormalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    colormap = get_voc_colormap()

    for idx in sample_indices:
        original_image, original_label = train_dataset.pull_item(idx)
        
        aug_sample = train_dataset[idx]
        aug_image = aug_sample['image']
        aug_label = aug_sample['label']

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        # Original image
        axs[0, 0].imshow(np.array(original_image).astype(np.uint8))
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis('off')

        # Original label
        original_label_rgb = visualize_label(np.array(original_label), colormap)
        axs[0, 1].imshow(original_label_rgb)
        axs[0, 1].set_title("Original Label")
        axs[0, 1].axis('off')

        # Augmented image
        denormalized_image = denormalize(aug_image.clone(), cfg.dataset.mean, cfg.dataset.std)
        denormalized_image = (denormalized_image.permute(1, 2, 0) * 255).clamp(0, 255).byte().numpy()
        axs[1, 0].imshow(denormalized_image)
        axs[1, 0].set_title("Augmented Image")
        axs[1, 0].axis('off')
        
        # Augmented label
        aug_label_rgb = visualize_label(aug_label.numpy(), colormap)
        axs[1, 1].imshow(aug_label_rgb)
        axs[1, 1].set_title("Augmented Label")
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"augmentation_sample_{idx}.png"))
        plt.close()

        print(f"Sample {idx}:")
        print(f"Original Image - Min: {np.min(original_image)}, Max: {np.max(original_image)}")
        print(f"Original Label - Min: {np.min(original_label)}, Max: {np.max(original_label)}")
        print(f"Augmented Image - Min: {torch.min(aug_image).item()}, Max: {torch.max(aug_image).item()}")
        print(f"Augmented Label - Min: {torch.min(aug_label).item()}, Max: {torch.max(aug_label).item()}")
        print("---")
        
    print(f"Augmentation samples saved to {output_dir}")