import os
from PIL import Image


from torch.utils.data import Dataset

def is_image(filename):
    return any(filename.endswith(ext) for ext in '.png')

def is_label(filename):
    return filename.endswith(".png")

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class MYDataset(Dataset):
    
    def __init__(self, root, split, transform):
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
        label_class_img = Image.open(label_file_path).convert('L')  


        sample = {'image': img, 'label': label_class_img}

        # 3. データ拡張を実施
        if self.transform:
            sample = self.transform(sample)


        return sample
