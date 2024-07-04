from typing import List, Dict, Optional, Union
import math

import torchvision.transforms.functional as F
from torch import Tensor

from augment import Cutout, solarize_add



# This function will apply to the RandAugment
# TODO: apply to the RA
def _apply_op(
    sample: Dict[str, Tensor],
    op_name: str,
    magnitude: float,
    interpolation: F.InterpolationMode,
    fill: Optional[List[float]]
):
    img, label = sample['image'], sample['label']

    if op_name == "ShearX":
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
        label = F.affine(
            label,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=F.InterpolationMode.NEAREST,
            fill=255,
            center=[0, 0],
        )
    elif op_name == "ShearY":
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
        label = F.affine(
            label,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=F.InterpolationMode.NEAREST,
            fill=255,
            center=[0, 0],
        )
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
        label = F.affine(
            label,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=F.InterpolationMode.NEAREST,
            shear=[0.0, 0.0],
            fill=255,
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
        label = F.affine(
            label,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=F.InterpolationMode.NEAREST,
            shear=[0.0, 0.0],
            fill=255,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
        label = F.rotate(label, magnitude, interpolation=F.InterpolationMode.NEAREST, fill=255)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
        # ラベルには適用しない
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
        # ラベルには適用しない
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
        # ラベルには適用しない
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
        # ラベルには適用しない
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
        # ラベルには適用しない
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
        # ラベルには適用しない
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
        # ラベルには適用しない
    elif op_name == "Equalize":
        img = F.equalize(img)
        # ラベルには適用しない
    elif op_name == "Invert":
        img = F.invert(img)
        # ラベルには適用しない
    elif op_name == "Identity":
        pass
    elif op_name == "Cutout":
        _, height, width = F.get_dimensions(img)
        cutout = Cutout(n_holes=1, img_size=height, patch_size=magnitude)
        img = cutout(img)
        label = cutout(label)
    elif op_name == "SolarizeAdd":
        img = solarize_add(image=img, addition=int(magnitude), threshold=128)
        # ラベルには適用しない
    elif op_name == "Hflip":
        img = F.hflip(img)
        label = F.hflip(label)
    elif op_name == "Vflip":
        img = F.vflip(img)
        label = F.vflip(label)
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    
    return {'image': img, 'label': label}

