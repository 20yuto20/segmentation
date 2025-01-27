import numpy as np
import os
import torch
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from augment import Cutout, Normalize, ToTensor, Normalize_Tensor, RandomCrop
from ra import RandAugmentSegmentation
from load_dataset.city import MYDataset
from load_dataset.voc import VOCDataset, datapath_list
# cfg.default.dataset_dir :  (SGE_LOCAL_DIR) + dataset/

from torch.utils.data._utils.collate import default_collate

def custom_collate_fn(batch):
    # バッチ内の画像形状を確認（デバッグ用）
    # for item in batch:
    #     print(f"Image shape: {item['image'].shape}")
    #     print(f"Label shape: {item['label'].shape}")
    
    max_h = max([item['image'].shape[1] for item in batch])
    max_w = max([item['image'].shape[2] for item in batch])
    
    for item in batch:
        # labelが2次元の場合 → 3次元 [1, H, W] に変換
        if len(item['label'].shape) == 2:
            item['label'] = item['label'].unsqueeze(0)
        # labelが1次元の場合 → [1, 1, W] に変換
        elif len(item['label'].shape) == 1:
            item['label'] = item['label'].unsqueeze(0).unsqueeze(0)
        
        # 画像・ラベルをバッチ内最大サイズに合わせてリサイズ
        if item['image'].shape[1] != max_h or item['image'].shape[2] != max_w:
            item['image'] = F.resize(item['image'], [max_h, max_w], interpolation=F.InterpolationMode.BILINEAR)
            item['label'] = F.resize(item['label'], [max_h, max_w], interpolation=F.InterpolationMode.NEAREST)
    
    return default_collate(batch)

class RandomScaleCrop(object):
    """
    画像とラベルを [scale_min, scale_max] の範囲でランダムスケールし、
    必要に応じてパディングしてから最終的に crop_size × crop_size に切り抜く。
    """
    def __init__(self, scale_min=0.5, scale_max=2.0, crop_size=473, ignore_label=255):
        """
        Args:
            scale_min (float): 拡大縮小の最小倍率
            scale_max (float): 拡大縮小の最大倍率
            crop_size (int): 最後に切り抜くサイズ (crop_size, crop_size)
            ignore_label (int): ラベルの無効値
        """
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.crop_size = crop_size
        self.ignore_label = ignore_label

    def __call__(self, sample):
        """
        Args:
            sample (dict): {'image': PIL or Tensor, 'label': PIL or Tensor}
        Returns:
            dict: {'image': PIL.Image, 'label': PIL.Image}
                  ※最終的に (crop_size, crop_size) に整形
        """
        img, label = sample['image'], sample['label']

        # 1) ランダムなスケール係数をサンプリング
        scale = random.uniform(self.scale_min, self.scale_max)

        # 画像の現在サイズを取得
        w, h = img.size  # (width, height)
        # 新しいサイズ
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 2) 画像とラベルをリサイズ
        # (画像にはバイリニア補間、ラベルにはニアレスト補間)
        img = F.resize(img, (new_h, new_w), interpolation=F.InterpolationMode.BILINEAR)
        label = F.resize(label, (new_h, new_w), interpolation=F.InterpolationMode.NEAREST)

        # 3) 必要に応じてパディング（スケール後に crop_size 以下になった部分をカバーする）
        pad_bottom = max(0, self.crop_size - new_h)
        pad_right  = max(0, self.crop_size - new_w)

        if pad_bottom > 0 or pad_right > 0:
            # 画像のパディング (RGBは0埋め)
            img = F.pad(img, (0, 0, pad_right, pad_bottom), fill=0)
            # ラベルは ignore_label で埋める
            label = F.pad(label, (0, 0, pad_right, pad_bottom), fill=self.ignore_label)

        # パディング後のサイズを更新
        w_padded, h_padded = img.size

        # 4) ランダムに crop_size × crop_size を切り抜き
        #    (切り抜き可能範囲を乱数で指定)
        x1 = random.randint(0, w_padded - self.crop_size)
        y1 = random.randint(0, h_padded - self.crop_size)

        img = F.crop(img, y1, x1, self.crop_size, self.crop_size)
        label = F.crop(label, y1, x1, self.crop_size, self.crop_size)

        return {'image': img, 'label': label}


def get_dataloader(cfg):
    train_transform = get_composed_transform(cfg, "train")
    val_transform = get_composed_transform(cfg, "val")
    test_transform = get_composed_transform(cfg, "test")

    if cfg.dataset.name == "voc":
        path_train = cfg.default.dataset_dir + "train_aug/"
        path_val = cfg.default.dataset_dir + "val/"
        path_test = cfg.default.dataset_dir + "test_2007/"

        print(f"load train from : {path_train} \n load validation from : {path_val} \n load test from : {path_test}")

        train_img_list, train_anno_list, val_img_list, val_anno_list, test_img_list, test_anno_list = datapath_list(
            path_train=path_train,
            path_val=path_val,
            path_test=path_test
        )
        
        train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train",
                                   transform=train_transform, img_size=cfg.dataset.resized_size)
        val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val",
                                 transform=val_transform, img_size=cfg.dataset.resized_size)
        test_dataset = VOCDataset(test_img_list, test_anno_list, phase="test",
                                  transform=test_transform, img_size=cfg.dataset.resized_size)

    else:
        # もし別のデータセットを使うならここに追記
        raise ValueError(f"Unsupported dataset name: {cfg.dataset.name}")

    print(f"train dataset len : {len(train_dataset)}")
    print(f"val dataset len : {len(val_dataset)}")
    print(f"test dataset len : {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.learn.batch_size, 
        num_workers=cfg.default.num_workers, 
        shuffle=True,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.learn.batch_size, 
        num_workers=cfg.default.num_workers, 
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn
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
            # if aug_name == "rcrop":
            #     transform_list.append(
            #         RandomCrop(size=cfg.dataset.resized_size, padding=cfg.augment.hp.rcrop_pad)
            #     )
            if aug_name == "rcrop":
                # 例：0.5～2.0倍にランダムスケールし、最後に crop_size=cfg.dataset.resized_size で切り抜き
                transform_list.append(
                    RandomScaleCrop(
                        scale_min=0.5,
                        scale_max=2.0,
                        crop_size=cfg.dataset.resized_size,
                        ignore_label=(cfg.dataset.ignore_label if "ignore_label" in cfg.dataset else 255)
                       )
                )

            elif  aug_name == "hflip":
                transform_list.append(
                    transforms.RandomApply(
                        [lambda x: {
                            'image': ImageOps.mirror(x['image']),
                            'label': ImageOps.mirror(x['label'])
                        }],
                        p=0.5
                    )
                )
            elif aug_name == "vflip":
                transform_list.append(
                    transforms.RandomApply(
                        [lambda x: {
                            'image': ImageOps.flip(x['image']),
                            'label': ImageOps.flip(x['label'])
                        }],
                        p=0.5
                    )
                )
            elif aug_name == "cutout":
                transform_list.append(
                    transforms.RandomApply(
                        [lambda x: {
                            'image': Cutout(n_holes=1, img_size=cfg.dataset.resized_size,
                                            patch_size=cfg.augment.hp.cutout_size)(x['image']),
                            'label': x['label']
                        }],
                        p=cfg.augment.hp.cutout_p
                    )
                )
            elif aug_name == "ra":
                transform_list.append(RandAugmentSegmentation(
                    cfg=cfg, num_ops=cfg.augment.ra.num_op, magnitude=cfg.augment.ra.magnitude))
            elif aug_name == "nan":
                pass
            elif aug_name == "gaussian_blur":
                transform_list.append(
                    transforms.RandomApply(
                        [lambda x: {
                            'image': F.gaussian_blur(x['image'], kernel_size=random.choice([3, 5, 7])),
                            'label': x['label']
                        }],
                        p=0.5
                    )
                )
            elif aug_name == "random_resize":
                transform_list.append(
                    lambda x: {
                        'image': F.resize(
                            x['image'],
                            size=[
                                int(x['image'].size[0] * random.uniform(0.5, 2.0)),
                                int(x['image'].size[1] * random.uniform(0.5, 2.0))
                            ],
                            interpolation=F.InterpolationMode.BILINEAR
                        ),
                        'label': F.resize(
                            x['label'],
                            size=[
                                int(x['label'].size[0] * random.uniform(0.5, 2.0)),
                                int(x['label'].size[1] * random.uniform(0.5, 2.0))
                            ],
                            interpolation=F.InterpolationMode.NEAREST
                        )
                    }
                )
            else:
                raise ValueError(f"Invalid Augment ... {aug_name}")

    # 学習時以外の共通・または学習/推論共通処理などここへ

    # 最後に正規化
    if cfg.dataset.name == "voc":
        transform_list.append(Normalize_Tensor(
            color_mean=cfg.dataset.mean, color_std=cfg.dataset.std))
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
    import os
    import random
    import torch
    import matplotlib.pyplot as plt
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
