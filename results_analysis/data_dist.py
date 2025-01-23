import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
# from augment import Normalize_Tensor, Resize  # Assuming Resize is defined in augment.py
import xml.etree.ElementTree as ET
import torch

# Assuming augment.py is in the same directory or a directory in your Python path
# If augment.py is located elsewhere, adjust the import path accordingly.
# For example, if augment.py is in a subdirectory named 'utils', use:
# from utils.augment import Normalize_Tensor, Resize

# Placeholder for Resize class (if not defined in augment.py or elsewhere)
class Resize:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, img, anno_class_img):
        # Resize the image
        resize_transform = transforms.Resize(self.img_size)
        img = resize_transform(img)

        # Resize the annotation image using nearest-neighbor interpolation
        resize_transform_label = transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST)
        anno_class_img = resize_transform_label(anno_class_img)
        
        return img, anno_class_img

def datapath_list(path_train, path_val, path_test):
    train_imgpath_template = os.path.join(path_train, 'image','%s.jpg')
    val_imgpath_template = os.path.join(path_val, 'image','%s.jpg')
    test_imgpath_template = os.path.join(path_test, 'image', '%s.jpg')
    
    train_annopath_template = os.path.join(path_train, 'label', '%s.png')
    val_annopath_template = os.path.join(path_val, 'label', '%s.png')
    test_annopath_template = os.path.join(path_test, 'label', '%s.png')
    
    train_id_names = os.path.join(path_train + 'trainaug.txt')
    val_id_names = os.path.join(path_val + 'val.txt')
    test_id_names = os.path.join(path_test + 'test.txt') 
    
    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (train_imgpath_template % file_id)  # 画像のパス
        anno_path = (train_annopath_template % file_id)  # アノテーションのパス
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (val_imgpath_template % file_id)  # 画像のパス
        anno_path = (val_annopath_template % file_id)  # アノテーションのパス
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    # テストデータの画像ファイルとアノテーションファイルへのパスリストを作成
    test_img_list = list()
    test_anno_list = list()

    for line in open(test_id_names):
        file_id = line.strip()
        img_path = (test_imgpath_template % file_id)
        anno_path = (test_annopath_template % file_id)
        test_img_list.append(img_path)
        test_anno_list.append(anno_path)
        
    return train_img_list, train_anno_list, val_img_list, val_anno_list, test_img_list, test_anno_list

class VOCDataset(Dataset):
    def __init__(self, img_list, anno_list, phase, transform, img_size):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, anno_class_img = self.pull_item(index)
        sample = {'image': img, 'label': anno_class_img}
        
        if self.transform:
            sample = self.transform(sample)
        return sample

    def pull_item(self, index):
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [高さ][幅][色RGB]

        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)   # [高さ][幅]

        resize_fn = Resize(self.img_size)
        img, anno_class_img = resize_fn(img, anno_class_img)

        return img, anno_class_img

def visualize_class_distribution(dataset, dataset_name):
    class_counts = {}
    label_names = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor',
        255: 'ignore'
    }
    for _, anno_path in dataset:
        with Image.open(anno_path) as img:
            unique_labels = np.unique(np.array(img))
            for label in unique_labels:
                label_name = label_names.get(label, str(label))
                if label_name in class_counts:
                    class_counts[label_name] += 1
                else:
                    class_counts[label_name] = 1

    # グラフの描画
    plt.figure(figsize=(8, 10))
    bars = plt.barh(list(class_counts.keys()), list(class_counts.values()), color='skyblue')
    plt.ylabel('Class Labels', fontsize=14)
    plt.xlabel('Number of Pixels', fontsize=14)
    plt.title(f'Class Distribution in {dataset_name}', fontsize=16)
    plt.yticks(list(class_counts.keys()), rotation=0, fontsize=12)
    
    # 各棒の上に数値を表示
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{int(width)}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_class_distribution.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    # データセットのパス（適宜変更してください）
    path_train = '/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc_aug/train_aug/'
    path_val = '/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc_aug/val/'
    path_test = '/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc_aug/test_2007/'
    
    # datapath_list関数を使ってリストを取得
    train_img_list, train_anno_list, val_img_list, val_anno_list, test_img_list, test_anno_list = datapath_list(path_train, path_val, path_test)

    # クラス分布の可視化
    visualize_class_distribution(zip(train_img_list, train_anno_list), 'Train Dataset')
    visualize_class_distribution(zip(val_img_list, val_anno_list), 'Validation Dataset')
    visualize_class_distribution(zip(test_img_list, test_anno_list), 'Test Dataset')
