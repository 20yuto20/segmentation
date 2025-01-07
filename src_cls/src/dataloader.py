import glob
import os
import random
from PIL import Image
import xml.etree.ElementTree as ET
import json

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

from utils.common import show_img
from augment import Cutout
from randaugment import RandAugment

class TinyImageNetDatasetLoader(Dataset):
    def __init__(self, root, split, img_size, transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.img_size = img_size
        
        self.images = []
        self.labels = []
        
        if split == 'train':
            self.data_dir = os.path.join(root, 'train')
            self._load_train_data()
        elif split == 'val':
            self.data_dir = os.path.join(root, 'val')
            self._load_val_data()
        elif split == 'test':
            self.data_dir = os.path.join(root, 'test')
            self._load_test_data()
            
    def _load_train_data(self):
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir, 'images')
            class_idx = self.class_to_idx[class_dir]
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.images.append(img_path)
                label = torch.zeros(200)
                label[class_idx] = 1
                self.labels.append(label)
                
    def _load_val_data(self):
        annotations_file = os.path.join(self.data_dir, 'val_annotations.txt')
        with open(annotations_file, 'r') as f:
            for line in f:
                img_name, class_dir = line.strip().split('\t')[:2]
                img_path = os.path.join(self.data_dir, 'images', img_name)
                class_idx = self.class_to_idx[class_dir]
                self.images.append(img_path)
                label = torch.zeros(200)
                label[class_idx] = 1
                self.labels.append(label)
                
    def _load_test_data(self):
        images_dir = os.path.join(self.data_dir, 'images')
        for img_name in os.listdir(images_dir):
            img_path = os.path.join(images_dir, img_name)
            self.images.append(img_path)
            # For test set, we'll use dummy labels
            self.labels.append(torch.zeros(200))

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        
        image = Image.open(image_path).convert("RGB")
        resize_fn = transforms.Resize((self.img_size, self.img_size))
        image = resize_fn(image)
        if self.transform is not None:
            image = self.transform(image)
            
        return {"image": image, "label": label}

    def __len__(self):
        return len(self.images)

    @property
    def class_to_idx(self):
        if not hasattr(self, '_class_to_idx'):
            self._class_to_idx = {}
            wnids_path = os.path.join(self.root, 'wnids.txt')
            with open(wnids_path, 'r') as f:
                for idx, line in enumerate(f):
                    self._class_to_idx[line.strip()] = idx
        return self._class_to_idx

class VOCDatasetLoader(Dataset):
    def __init__(self, root, year, image_set, img_size, transform=None):
        super().__init__()
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transform = transform
        self.img_size = img_size
        
        self.images = []
        self.labels = []
        
        self._load_dataset()
        
    def _load_dataset(self):
        image_dir = os.path.join(self.root, f'VOC{self.year}', 'JPEGImages')
        annotation_dir = os.path.join(self.root, f'VOC{self.year}', 'Annotations')
        image_set_file = os.path.join(self.root, f'VOC{self.year}', 'ImageSets', 'Main', f'{self.image_set}.txt')
        
        with open(image_set_file, 'r') as f:
            for line in f:
                image_id = line.strip()
                image_path = os.path.join(image_dir, f'{image_id}.jpg')
                annotation_path = os.path.join(annotation_dir, f'{image_id}.xml')
                
                if os.path.exists(image_path) and os.path.exists(annotation_path):
                    self.images.append(image_path)
                    self.labels.append(self._parse_annotation(annotation_path))
    
    def _parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        classes = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in classes:
                classes.append(class_name)
        
        label = torch.zeros(len(self.class_to_idx))
        for cls in classes:
            label[self.class_to_idx[cls]] = 1
        
        return label

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        
        image = Image.open(image_path).convert("RGB")
        resize_fn = transforms.Resize((self.img_size, self.img_size))
        image = resize_fn(image)
        if self.transform is not None:
            image = self.transform(image)
            
        return {"image": image, "label": label}

    def __len__(self):
        return len(self.images)

    @property
    def class_to_idx(self):
        return {cls: idx for idx, cls in enumerate(self.classes)}

    @property
    def classes(self):
        return ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                'train', 'tvmonitor']

def get_dataloader(cfg):
    def worker_init_fn(worker_id):
        random.seed(worker_id+cfg.default.seed)

    dataset_path = f"{cfg.default.dataset_dir}"
    train_transform = get_composed_transform(cfg, "train")
    test_transform = get_composed_transform(cfg, "test")

    if cfg.dataset.name == "tiny_imagenet":
        train_dataset = TinyImageNetDatasetLoader(dataset_path, 'train', cfg.dataset.resized_size, train_transform)
        val_dataset = TinyImageNetDatasetLoader(dataset_path, 'val', cfg.dataset.resized_size, test_transform)
        test_dataset = TinyImageNetDatasetLoader(dataset_path, 'test', cfg.dataset.resized_size, test_transform)
    else:
        train_dataset = VOCDatasetLoader(dataset_path, '2012', 'train', cfg.dataset.resized_size, train_transform)
        val_dataset = VOCDatasetLoader(dataset_path, '2012', 'val', cfg.dataset.resized_size, test_transform)
        test_dataset = VOCDatasetLoader(dataset_path, '2007', 'test', cfg.dataset.resized_size, test_transform)

    print("train_data: {}".format(len(train_dataset)))
    print("val_data: {}".format(len(val_dataset)))
    print("test_data: {}".format(len(test_dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.learn.batch_size,
        num_workers=cfg.default.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.learn.batch_size,
        num_workers=cfg.default.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.learn.batch_size,
        num_workers=cfg.default.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    if cfg.save.img:
        show_img(cfg, train_loader)
    
    return train_loader, val_loader, test_loader

 
def get_composed_transform(cfg, phase):
    transform_list = []

    if phase == "train":

        for aug_name in cfg.augment.name:
            if aug_name == "rcrop":
                transform_list.append(
                    transforms.RandomCrop(size=cfg.dataset.resized_size, padding=cfg.augment.hp.rcrop_pad)
                )

            elif aug_name == "hflip":
                transform_list.append(
                    transforms.RandomHorizontalFlip(p=0.5)
                )
            
            elif aug_name == "vflip":
                transform_list.append(
                    transforms.RandomVerticalFlip(p=0.5)
                )

            elif aug_name == "cutout":
                transform_list.append(
                    transforms.RandomApply(
                        [Cutout(n_holes=1, img_size=cfg.dataset.resized_size, patch_size=cfg.augment.hp.cutout_size)],
                        p=cfg.augment.hp.cutout_p
                    )
                )
                
            elif aug_name == "ra":
                transform_list.append(
                    RandAugment(cfg=cfg, num_ops=cfg.augment.ra.num_op, magnitude=cfg.augment.ra.magnitude)
                )

            elif aug_name == "nan":
                pass
                
            else:
                    raise ValueError (f"Invalid Augment ... {aug_name}")
       
    transform_list = transform_list + [
        transforms.ToTensor(),
        transforms.Normalize(cfg.dataset.mean, cfg.dataset.std)
        ]
        
    transform_list = transforms.Compose(transform_list)

    return transform_list


        
# AffinityDataset クラスを VOC 用に修正
class AffinityDataset():
    def __init__(self, cfg, name):
        self.cfg = cfg.copy()
        self.cfg.augment.dynamic=False
        self.cfg.augment.name=["rand"]
        self.cfg.augment.rand.weight="single"
        self.cfg.augment.rand.num_op=1
        self.cfg.augment.rand.single=name
        self.size=cfg.dataset.resized_size

    def dataloader(self):
        train_transform, test_transform = get_composed_transform(self.cfg)

        dataset_path = f"{self.cfg.default.dataset_dir}"+ f"{self.cfg.dataset.name}" 
        # use train_transform for val dataset
        val_dataset = VOCDatasetLoader(dataset_path, '2012', 'val', self.cfg.dataset.resized_size, train_transform)

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.learn.batch_size,
            num_workers=self.cfg.default.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )
    
        return val_loader


# 何も変換なしのval loaderとtransfromを返す
def val_loader_transform(cfg):
    cfg = cfg.copy()
    cfg.augment.dynamic=False
    train_transform, test_transform = get_composed_transform(cfg)
    print("transform")
    print(train_transform)

    dataset_path = f"{cfg.default.dataset_dir}"+ f"{cfg.dataset.name}" 
    # use train_transform for val dataset
    val_dataset = VOCDatasetLoader(dataset_path, "val", cfg.dataset.resized_size, test_transform)

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.learn.batch_size,
        num_workers=cfg.default.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )

    return val_loader, train_transform
