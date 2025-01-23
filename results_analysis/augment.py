import torch
import torchvision.transforms.functional as F
from PIL import Image

class Normalize_Tensor(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.to_tensor(image)
        image = F.normalize(image, self.mean, self.std)
        label = torch.from_numpy(np.array(label))
        return {'image': image, 'label': label}

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        image = image.resize((self.size, self.size), Image.BILINEAR)
        label = label.resize((self.size, self.size), Image.NEAREST)
        return image, label
