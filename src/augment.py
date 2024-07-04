import numpy as np
import numbers
import math
import random
import torch
import torchvision.transforms.functional as F

from PIL import ImageOps, Image

class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
       
        return {'image': img,
                'label': mask}

class ToTensor(object):
    def __call__(self, sample):
       
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}

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
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=255)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size # target size
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask}
    
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

