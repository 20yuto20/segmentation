import numpy as np
import os
from PIL import Image, ImageOps, ImageFilter

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from augment import RandomCrop, Normalize, ToTensor
import sys




def is_image(filename):
    return any(filename.endswith(ext) for ext in '.png')

def is_label(filename):
    # return filename.endswith("_s.png")
    return filename.endswith(".png")

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class MYDataset(Dataset):
    
    # split : phase (train, val, test)
    # root : datsetがあるdirのパス
    def __init__(self, root, split, transform):

        # # detect the dir dynamically
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # parent_dir = os.path.dirname(current_dir)

        # self._base_dir = os.path.join(parent_dir, 'dataset', 'CityScapes')

        self._base_dir = root
        
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
        # label_class_img = Image.open(label_file_path).convert('L')  
        label_class_img = Image.open(label_file_path)
        print(label_class_img.mode)  
        image_array = np.array(label_class_img)
        # ピクセルの最大値と最小値を取得
        max_pixel_value = image_array.max()
        min_pixel_value = image_array.min()

        print(max_pixel_value)  
        print(min_pixel_value)
        sample = {'image': img, 'label': label_class_img}

        # 3. データ拡張を実施
        return self.transform(sample)
    
# デバッグ用，手元にdatsetがない時に適当なデータセットを使用
class RandomDataset(Dataset):
    def __init__(self, num_samples, transform=None):
        self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # ランダムなRGB画像（128x128）を生成
        img = Image.fromarray(np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8))
        
        # ランダムなラベル画像（128x128）を生成
        label = Image.fromarray(np.random.randint(0, 3, (128, 128), dtype=np.uint8))
        
        sample = {'image': img, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

def get_dataloader(cfg):
    # configファイルにあるデータセットを読み出すために dataset dir のpathをセット
    dataset_path = os.path.join(cfg.default.dataset_dir, cfg.dataset.name)
    
    # データ拡張を設定
    train_transform = transforms.Compose([                          
        RandomCrop((cfg.dataset.resized_size, cfg.dataset.resized_size)), 
        Normalize(cfg.dataset.mean, cfg.dataset.std),
        ToTensor(),
    ])

    test_transform = transforms.Compose([
        Normalize(cfg.dataset.mean, cfg.dataset.std),
        ToTensor(),
    ])

    # データセットの作成
    train_dataset = MYDataset(dataset_path, split='train', transform=train_transform)
    val_dataset = MYDataset(dataset_path, split='val', transform=test_transform)
    test_dataset = MYDataset(dataset_path, split='test', transform=test_transform)

    # train_dataset = RandomDataset(128, train_transform)
    # val_dataset = RandomDataset(128, test_transform)
    # test_dataset = RandomDataset(128, test_transform)

    # データローダーの作成
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.learn.batch_size, 
        num_workers=cfg.default.num_workers, 
        shuffle=True,
        pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        num_workers=0, 
        shuffle=False,
        pin_memory=True
        )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        num_workers=0, 
        shuffle=False,
        pin_memory=True
        )

    return train_loader, val_loader, test_loader


