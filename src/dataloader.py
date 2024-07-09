import numpy as np
import os
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from augment import Cutout
from ra import RandAugmentSegmentation

def is_image(filename):
    return any(filename.endswith(ext) for ext in '.png')

def is_label(filename):
    return filename.endswith(".png")

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class MYDataset(Dataset):
    
    def __init__(self, root, split, transform):
<<<<<<< HEAD

        # # detect the dir dynamically
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # parent_dir = os.path.dirname(current_dir)

        # self._base_dir = os.path.join(parent_dir, 'dataset', 'CityScapes')

        self._base_dir = root
        
=======
        self._base_dir = root
>>>>>>> 8a19810 (polynomialスケジューラーとsingleで制御できるロジックを追加)
        self.split = split
        self.images_root = os.path.join(self._base_dir, split, 'rgb/')
        self.labels_root = os.path.join(self._base_dir, split, 'label/')
        
        self.filenames = [image_basename(f)
            for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames.sort()
        self.filenamesGt = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_label(f)]
        
        self.filenamesGt.sort()
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
         # 1. 画像読み込み
        image_file_path = self.filenames[index]+ '.png'
        image_file_path = os.path.join(self._base_dir, self.split, 'rgb/', image_file_path)
        img = Image.open(image_file_path).convert('RGB')

        # 2. アノテーション画像読み込み
        label_file_path = self.filenamesGt[index]+ '.png'
        label_file_path = os.path.join(self._base_dir, self.split, 'label/', label_file_path)
        label_class_img = Image.open(label_file_path).convert('L')  
        print(label_class_img.mode)  
        image_array = np.array(label_class_img)
        # ピクセルの最大値と最小値を取得
        max_pixel_value = image_array.max()
        min_pixel_value = image_array.min()

        print(max_pixel_value)  
        print(min_pixel_value)
        sample = {'image': img, 'label': label_class_img}

        # 3. データ拡張を実施
        if self.transform:
            sample = self.transform(sample)

        return sample


def get_dataloader(cfg):
    cur_dir = Path(__file__).parent.parent
    dataset_name = cfg.dataset.name
    dataset_path = str(cur_dir / "dataset" / dataset_name)
    
    train_transform = get_composed_transform(cfg, "train")
    val_transform = get_composed_transform(cfg, "val")
    test_transform = get_composed_transform(cfg, "test")

    train_dataset = MYDataset(dataset_path, split='train', transform=train_transform)
    val_dataset = MYDataset(dataset_path, split='val', transform=val_transform)
    test_dataset = MYDataset(dataset_path, split='test', transform=test_transform)

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

    return train_loader, val_loader, test_loader

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
    
    transform_list.append(
        lambda x: {'image': transforms.ToTensor()(x['image']),
                   'label': transforms.ToTensor()(x['label'])}
    )
    transform_list.append(
        lambda x: {'image': transforms.Normalize(cfg.dataset.mean, cfg.dataset.std)(x['image']),
                   'label': x['label']}
    )
        
    return transforms.Compose(transform_list)
