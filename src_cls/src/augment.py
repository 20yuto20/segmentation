import glob
import os
import numpy as np
from PIL import Image
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

import sys

### Designed for use with PIL images (before ToTensor)


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
    

# Apply gaussian noise patch
class PatchGaussian(object):
    def __init__(self, patch_size, max_scale):
        self.patch_size = patch_size
        self.max_scale = max_scale

    def __call__(self, img):
        img = np.array(img)
       
        patch_size = np.array(self.patch_size)
        scale = random.uniform(0, 1) * self.max_scale
        # torch.randnはガウス分布を生成　
        gaussian = np.random.normal(loc = 0.0, scale = scale, size = img.shape) * 255

        image_plus_gaussian = np.clip((img + gaussian), 0, 255)
        # create mask and apply patch
        patch_mask = self._get_patch_mask(img, patch_size)
        #patch_mask = np.repeat(patch_mask[:,:,np.newaxis], img.shape[-1], axis=2)
        patch_mask = np.tile(patch_mask[:,:,np.newaxis], 3)
        
        # torch.where(condition, x, y)  True -> x, False -> y
        img = np.where(patch_mask, image_plus_gaussian, img)
        img = img.astype(np.uint8)
        return Image.fromarray(img)


    def _get_patch_mask(self, image, patch_size):
        image_size = image.shape[0]
    
        # randomly sample location in the image 
        x = np.random.randint(low=0, high=image_size, size=(1,))
        y = np.random.randint(low=0, high=image_size, size=(1,))
        x = x.astype(float)
        y = y.astype(float)

        startx = x - np.floor(patch_size / 2)
        starty = y - np.floor(patch_size / 2)
        endx = x + np.ceil(patch_size / 2)
        endy = y + np.ceil(patch_size / 2)

        # 画像の範囲超えないように
        startx = np.maximum(startx, 0)
        starty = np.maximum(starty, 0)
        endx = np.minimum(endx, image_size)
        endy = np.minimum(endy, image_size)

        lower_pad = image_size - endy
        upper_pad = starty
        right_pad = image_size - endx
        left_pad = startx

        padding_dims = (
            ((int(upper_pad), int(lower_pad)), (int(left_pad), int(right_pad)))
        )
        # create mask
        mask = np.pad(np.zeros((int(endy - starty), int(endx - startx))), padding_dims, constant_values=1)
        
        # Gaussian PatchすべきpatchをTrueで返す
        return mask == 0
    

# Load Dataset for Mix
class DatasetLoaderMix(Dataset):
    def __init__(self, root, phase, transform=None):
        super().__init__()
        self.transform = transform
        self.image_paths = []
        self.image_labels = []
        self.class_name = os.listdir(os.path.join(root, phase))
        self.class_name.sort()
        for i, x in enumerate(self.class_name):
            temp = sorted(glob.glob(os.path.join(root, phase, x, "*")))
            self.image_labels.extend([i] * len(temp))
            self.image_paths.extend(temp)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
            
        return {"image": image, "label": self.image_labels[index]}

    def __len__(self):
        return len(self.image_paths)

# MixUp for ONLY img (NO label mixing), 
### NOT Effective Augmentation, just for experiment
class MixImage():
    def __init__(self, alpha, dataset_loader):
        self.alpha = alpha
        self.dataset_loader = dataset_loader
    
    def __call__(self, img):        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            
        else:
            lam = 1
        
        n_dataset = len(self.dataset_loader)
        index = random.randint(0, n_dataset - 1)
        img2 = self.dataset_loader[index]
        img2 = img2["image"]
        img = np.array(img)
        img2 = np.array(img2)
        img = lam * img + (1 - lam) * img2
        img = img.astype(np.uint8)
        
        return Image.fromarray(img)
    

def solarize_add(image, addition=0, threshold=128):
    image_array = np.array(image, dtype=np.int64)

    added_image = image_array + addition
    clipped_image = np.clip(added_image, 0, 255)

    # 指定された閾値未満のピクセル値の領域に対して加算, クリップ
    result_image = np.where(image_array < threshold, clipped_image, image_array)
    result_image = result_image.astype(np.uint8)

    return Image.fromarray(result_image)


# 
class RandomDownSize(object):
    def __init__(self, orig_size, target_size):
        self.patch_size = int(orig_size / target_size)
        self.target_size = target_size

    def __call__(self, img):
        img = np.array(img)
        mini_img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        for i in range(self.target_size):
            for j in range(self.target_size):
                start_row, start_col = i * self.patch_size, j * self.patch_size
                end_row, end_col = start_row + self.patch_size, start_col + self.patch_size
                row_idx = random.randint(start_row, end_row - 1)
                col_idx = random.randint(start_col, end_col - 1)
                mini_img[i, j] = img[row_idx, col_idx]

        return Image.fromarray(mini_img)





