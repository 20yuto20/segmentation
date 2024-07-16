import numpy as np
import numbers
import random
import torch
import torchvision.transforms.functional as F
from torchvision import transforms

from PIL import ImageOps, Image

class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        print(img.size())
        print(mask.size())
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.int64)  # ラベルはint64型に変換
        img /= 255.0
        img -= self.mean
        img /= self.std
       
        return {'image': img, 'label': mask}


class Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, sample):
        img = sample['image']
        anno_class_img = sample['label']

        # PIL画像をTensorに。大きさは最大1に規格化される
        img = transforms.functional.to_tensor(img)

        # 色情報の標準化
        img = transforms.functional.normalize(
            img, self.color_mean, self.color_std)

        # アノテーション画像をNumpyに変換
        anno_class_img = np.array(anno_class_img)  # [高さ][幅]

        #### VOCの時この処理に注意　#####
        # 'ambigious'には255が格納されているので、0の背景にしておく
        index = np.where(anno_class_img == 255)
        anno_class_img[index] = 0

        # アノテーション画像をTensorに
        anno_class_img = torch.from_numpy(anno_class_img)

        return {'image': img, 'label': anno_class_img}
    


class ToTensor(object):
    def __call__(self, sample):
       
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img, 'label': mask}
    


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']

        if self.padding > 0:
            img = F.pad(img, self.padding, fill=0)
            mask = F.pad(mask, self.padding, fill=255)

        # クロップ位置を乱数で固定
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.size)
    
        # 画像とラベルを同じ位置でクロップ    
        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        return {'image': img, 'label': mask}
    
    
def resize(sample, size, interpolation="bilinear", max_size=None, antialias=None):
    img, label = sample['image'], sample['label']
    
    def _resize_img(img, size, interpolation, max_size, antialias):
        return F.resize(img, size, interpolation=interpolation, max_size=max_size, antialias=antialias)
    
    def _resize_label(label, size, interpolation, max_size, antialias):
        return F.resize(label.unsqueeze(0), size, interpolation="nearest").squeeze(0)
    
    if isinstance(size, (int, float)):
        size = [size, size]
    
    img_resized = _resize_img(img, size, interpolation, max_size, antialias)
    label_resized = _resize_label(label, size, interpolation, max_size, antialias)
    
    return {'image': img_resized, 'label': label_resized}


class Resize(object):
    def _init__(self, size, interpolation="bilinear", max_size=None, antialias=None):
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def __call__(self, sample):
        return resize(sample, self.size, self.interpolation, self.max_size, self.antialias)


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, img_size, patch_size):
        self.n_holes = n_holes
        self.length = img_size * patch_size
        

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        h = img.size(1)
        w = img.size(2)
        """
        _, h, w = F.get_dimensions(img)

        mask = np.ones((h, w), np.uint8)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h).astype(int)
            y2 = np.clip(y + self.length // 2, 0, h).astype(int)
            x1 = np.clip(x - self.length // 2, 0, w).astype(int)
            x2 = np.clip(x + self.length // 2, 0, w).astype(int)
            
            mask[y1: y2, x1: x2] = 0

        if torch.is_tensor(img):
            mask = torch.from_numpy(mask)      
            mask = mask.expand_as(img)

        else:
            mask = mask[:,:,np.newaxis]
            mask = np.tile(mask, 3)
 
        img = img * mask

        return Image.fromarray(img)
    
def solarize_add(image, addition=0, threshold=128):
    image_array = np.array(image, dtype=np.int64)

    added_image = image_array + addition
    clipped_image = np.clip(added_image, 0, 255)

    # 指定された閾値未満のピクセル値の領域に対して加算, クリップ
    result_image = np.where(image_array < threshold, clipped_image, image_array)
    result_image = result_image.astype(np.uint8)

    return Image.fromarray(result_image)

