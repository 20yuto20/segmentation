import os
from PIL import Image, ImageOps, ImageFilter

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from augment import RandomCrop, Normalize, ToTensor, Resize




def is_image(filename):
    return any(filename.endswith(ext) for ext in '.png')

def is_label(filename):
    # return filename.endswith("_s.png")
    return filename.endswith(".png")

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class MYDataset(Dataset):
    
    # split : phase
    def __init__(self, split, transform):

        # detect the dir dynamically
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)

        self._base_dir = os.path.join(parent_dir, 'dataset', 'CitySpaces')
        
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
        sample = {'image': img, 'label': label_class_img}

        # 3. データ拡張を実施
        return self.transform(sample)

def get_dataloader():
    # データ拡張を設定
    # TODO: These image sizes shoulf be managed by YAML
    transform = transforms.Compose([                  
        Resize((128,128)),        
        RandomCrop((128,128)), 
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensor(),
    ])

    test_transform = transforms.Compose([
        Resize((128,128)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensor(),
    ])

    # データセットの作成
    train_dataset = MYDataset(split='train', transform=transform)
    val_dataset = MYDataset(split='val', transform=test_transform)

    # データローダーの作成
    batch_size = 5
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=False,pin_memory=True)

    return train_loader, val_loader


