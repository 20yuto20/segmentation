import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img_path = 'path/to/image.jpg'
img = Image.open(img_path).convert('RGB')

aug_name = [
    'AutoContrast',
    'posterize',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'TranslateX',
    'TranslateY',
    'Rotate',
    'Sharpness',
    'ShearX',
    'ShearY',
    'Brightness',
    'HorizontalFlip',
    'VerticalFlip',
    'No Augmentation',
    'cutout',
    'Randome Crop',
]