from typing import List, Dict, Tuple, Optional
import math

import torch
from torch import Tensor, nn
import torchvision.transforms.functional as F

from augment import Cutout, solarize_add



def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: F.InterpolationMode, fill: Optional[List[float]]
):
    # shear degree
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    # magnitude: 0 - 14.5..
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        past = img
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    elif op_name == "Cutout":
        _, height, width = F.get_dimensions(img)
        cutout = Cutout(n_holes=1, img_size=height, patch_size=magnitude)
        img = cutout(img)
    elif op_name == "SolarizeAdd":
        img = solarize_add(image=img, addition=int(magnitude), threshold=128)
    elif op_name == "Hflip":
        img = F.hflip(img)
    elif op_name == "Vflip":
        img = F.vflip(img)
    
    
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img



class DefineAugmentSpace(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def _ra_augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        space_dict = {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor([0.0]), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 10.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 10.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.1, 1.9, num_bins), True),
            "Color": (torch.linspace(0.1, 1.9, num_bins), True),
            "Contrast": (torch.linspace(0.1, 1.9, num_bins), True),
            "Sharpness": (torch.linspace(0.1, 1.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "SolarizeAdd": (torch.linspace(0, 110.0, num_bins), False),
            "Invert": (torch.tensor([0.0]), False),
            "AutoContrast": (torch.tensor([0.0]), False),
            "Equalize": (torch.tensor([0.0]), False),
        }
        if image_size[0] > 100:
            space_dict["TranslateX"] = (torch.linspace(0.0, 100.0, num_bins), True)
            space_dict["TranslateY"] = space_dict["TranslateX"]

        return space_dict


    def _ra_wide_augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor([0.0]), False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.1, 1.9, num_bins), True),
            "Color": (torch.linspace(0.1, 1.9, num_bins), True),
            "Contrast": (torch.linspace(0.1, 1.9, num_bins), True),
            "Sharpness": (torch.linspace(0.1, 1.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor([0.0]), False),
            "Equalize": (torch.tensor([0.0]), False),
            "Vflip":(torch.tensor([0.0]), False)
        }

    def _original_augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        space_dict = {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor([0.0]), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 10, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 10, num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.1, 1.9, num_bins), True),
            "Color": (torch.linspace(0.1, 1.9, num_bins), True),
            "Contrast": (torch.linspace(0.1, 1.9, num_bins), True),
            "Sharpness": (torch.linspace(0.1, 1.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor([0.0]), False),
            "Equalize": (torch.tensor([0.0]), False),
            ###
            "Cutout": (torch.linspace(0.0, 0.5, num_bins), False),
            "SolarizeAdd": (torch.linspace(0, 110.0, num_bins), False),
            "Invert": (torch.tensor([0.0]), False),
            ###
            "Hflip":(torch.tensor([0.0]), False),
            "Vflip":(torch.tensor([0.0]), False),
        }
        
        return space_dict

    def _jda_augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        space_dict = {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor([0.0]), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 10.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 10.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.1, 1.9, num_bins), True),
            "Color": (torch.linspace(0.1, 1.9, num_bins), True),
            "Contrast": (torch.linspace(0.1, 1.9, num_bins), True),
            "Sharpness": (torch.linspace(0.1, 1.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor([0.0]), False),
            "Equalize": (torch.tensor([0.0]), False),
            "Invert": (torch.tensor([0.0]), False),
        }
        if image_size[0] > 100:
            space_dict["TranslateX"] = (torch.linspace(0.0, 100.0, num_bins), True)
            space_dict["TranslateY"] = space_dict["TranslateX"]

        return space_dict

