from typing import List, Dict, Optional, Union, Tuple
import math
import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import ImageOps, Image
import torch
import torchvision.transforms.functional as F
from torch import Tensor, nn

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
    ## debug and test code for Translate X
    # elif op_name == "TranslateX":
    #     # 画像の幅の30%の移動を適用
    #     width = img.size[-1]  # 画像の幅を取得
    #     fixed_magnitude = int(0.3 * width)  # 幅の30%を計算

    #     img = F.affine(
    #         img,
    #         angle=0.0,
    #         translate=[fixed_magnitude, 0],  # fixed_magnitude を使用
    #         scale=1.0,
    #         interpolation=interpolation,
    #         shear=[0.0, 0.0],
    #         fill=fill,
    #     )
    #     label = F.affine(
    #         label,
    #         angle=0.0,
    #         translate=[fixed_magnitude, 0],  # 同じ値を使用
    #         scale=1.0,
    #         interpolation=F.InterpolationMode.NEAREST,
    #         shear=[0.0, 0.0],
    #         fill=255,
    #     )
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
    # # debug and test for Translate Y
    # elif op_name == "TranslateY":
    #     height = img.size[-2]
    #     fixed_mag = int(height * 0.3)
    #     img = F.affine(
    #         img,
    #         angle=0.0,
    #         translate=[0, fixed_mag],
    #         scale=1.0,
    #         interpolation=interpolation,
    #         shear=[0.0, 0.0],
    #         fill=fill,
    #     )
    #     label = F.affine(
    #         label,
    #         angle=0.0,
    #         translate=[0, fixed_mag],
    #         scale=1.0,
    #         interpolation=F.InterpolationMode.NEAREST,
    #         shear=[0.0, 0.0],
    #         fill=255,
    #     )
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
        magnitude = 2.0
        img = F.adjust_brightness(img, 1.0 + magnitude)
        # # print(f"Applied {op_name} with magnitude {magnitude}")  
        # brightness_factor = max(0.5, min(1.5, 1.0 + magnitude))  # 0.5から1.5の範囲に制限
        # img = F.adjust_brightness(img, brightness_factor)
        # ラベルには適用しない
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
        # ラベルには適用しない
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
        # ラベルには適用しない
    elif op_name == "Sharpness":
        # magnitude = 5.0
        img = F.adjust_sharpness(img, 1.0 + magnitude)
        # ラベルには適用しない
    elif op_name == "Posterize":
        # magnitude = 2.0
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
    # FIXME: NOT applied
    elif op_name == "Invert":
        
        # original_pix = np.array(img)
        # print(f"orignal pixel: {original_pix}")
        
        img = ImageOps.invert(img)
        
        # inverted_pix = np.array(img)
        # print(f"inverted pix: {inverted_pix}")
        

        # pix_diff = np.abs(original_pix - inverted_pix)
        # print(f"Average pixel difference: {np.mean(pix_diff)}")
        
        # output_dir = "./output_debug_invert"
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        
        # # ヒストグラムを保存
        # plt.figure(figsize=(15, 5))
        # plt.subplot(1, 2, 1)
        # plt.title('Histogram of Original Image')
        # plt.hist(original_pix.flatten(), bins=256, color='blue', alpha=0.5, label='Original')
        # plt.xlabel('Pixel Value')
        # plt.ylabel('Frequency')
        # plt.legend()
        # original_hist_path = os.path.join(output_dir, 'original_histogram.png')
        # plt.savefig(original_hist_path)

        # plt.subplot(1, 2, 2)
        # plt.title('Histogram of Inverted Image')
        # plt.hist(inverted_pix.flatten(), bins=256, color='red', alpha=0.5, label='Inverted')
        # plt.xlabel('Pixel Value')
        # plt.ylabel('Frequency')
        # plt.legend()
        # inverted_hist_path = os.path.join(output_dir, 'inverted_histogram.png')
        # plt.savefig(inverted_hist_path)

        # plt.show()
        # ラベルには適用しない
    elif op_name == "Identity":
        pass
    elif op_name == "Cutout":
        _, height, width = F.get_dimensions(img)
        cutout = Cutout(n_holes=1, img_size=min(height, width), patch_size=magnitude)
        sample = cutout({'image': img, 'label': label})
        img, label = sample['image'], sample['label']
    elif op_name == "Cutout_img":
        _, height, width = F.get_dimensions(img)
        cutout = Cutout(n_holes=1, img_size=min(height, width), patch_size=magnitude)
        sample = cutout({'image' : img})
        img = sample['image']
    elif op_name == "SolarizeAdd":
        img = solarize_add(image=img, addition=int(magnitude), threshold=128)
        # ラベルには適用しない
    # elif op_name == "Hflip":
    #     img = ImageOps.mirror(img)
    #     label = ImageOps.mirror(label)
    # elif op_name == "Vflip":
    #     img = F.vflip(img)
    #     label = F.vflip(label)
        
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    
    return {'image': img, 'label': label}


# copy and paste from original code
# TODO: if necessary, change this code into segmentaion version
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
            "AutoContrast": (torch.tensor([0.0]), False),
            "Equalize": (torch.tensor([0.0]), False),
            "Cutout": (torch.linspace(0.0, 0.5, num_bins), False),
            "SolarizeAdd": (torch.linspace(0, 110.0, num_bins), False),
            "Invert": (torch.tensor([0.0]), False),
            "Hflip":(torch.tensor([0.0]), False),
            "Vflip":(torch.tensor([0.0]), False)
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
            "Equalize": (torch.tensor([0.0]), False)
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
