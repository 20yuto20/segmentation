import random
import torch
import os
import torchvision.transforms.functional as F
import numpy as np
import pandas as pd
from torch import Tensor, nn
from omegaconf import OmegaConf
from typing import List, Dict, Optional, Tuple

from aug_meta import DefineAugmentSpace, _apply_op
from set_cfg import override_original_config

# TODO: w/ affinity dataloader.py & affinity.pyにRandAugmentSegmentationクラスを元のコードを参考にして継承させる

def reset_cfg(cfg, init: bool):
    if init:
        if cfg.augment.ra.init_epoch is None:
            raise ValueError("Error... set num of init phase epoch")
        print(f"Set cfg for init phase, num of init phase epoch ...{cfg.augment.ra.init_epoch}")

        if cfg.augment.name[0] == "w_ra":
            cfg.augment.ra.warmup_ra = True
            print("Apply warmup RA")
    
        cfg.augment.name = ["nan"]
        cfg.save.affinity = True

    else:
        print("Set cfg for main phase")
        cfg.augment.name=["ra"]
        cfg.augment.ra.weight="affinity"
        cfg.save.affinity=False

        if cfg.augment.ra.warmup_ra:
            print("Set for warmup RA, weight is random")
            cfg.augment.ra.weight="random"

    override_original_config(cfg)
    print(OmegaConf.to_yaml(cfg))
    return cfg

class RandAugmentSegmentation(torch.nn.Module):
    def __init__(
        self,
        cfg,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: F.InterpolationMode = F.InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

        self.weight_type = cfg.augment.ra.weight
        self.softmax_t = cfg.augment.ra.softmax_t

        self.single = cfg.augment.ra.single

        self.space = DefineAugmentSpace()
        
        if self.cfg.augment.ra.space == "ra":
            self.op_meta = self.space._ra_augmentation_space(self.num_magnitude_bins, (32, 32))
        elif self.cfg.augment.ra.space == "jda":
            self.op_meta = self.space._jda_augmentation_space(self.num_magnitude_bins, (32, 32))

        self.weight = {key: 0 for key in self.op_meta.keys()}

        if "ra" in self.cfg.augment.name:
            if self.weight_type == "affinity":
                self.metrics_value = self.get_metrics_values(self.weight)

        self.weight = self.get_weight(self.weight)

        self.iden_rate = float(1 / len(self.op_meta))

        self.count = 0
        self.count_dict = {key: 0 for key in self.op_meta.keys()}

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.count += 1
        img, label = sample['image'], sample['label'] # apply ra to both image and label
        
        fill = self.fill
        channels, height, width = F.get_dimensions(img)

        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

         # aug spaceの設定
        if self.cfg.augment.ra.space == "ra":
            op_meta = self.space._ra_augmentation_space(self.num_magnitude_bins, (height, width))
        elif self.cfg.augment.ra.space == "jda":
            op_meta = self.space._jda_augmentation_space(self.num_magnitude_bins, (height, width))

        # 指定された回数だけ拡張操作を適用
        for _ in range(self.num_ops):
            # 重みに基づいて拡張操作を選択
            op_index = torch.multinomial(torch.tensor(list(self.weight.values())), 1, replacement=True).item()
            op_name = list(self.op_meta.keys())[op_index]
            # print(f"Selected operation: {op_name}")
            mag_range, signed = self.op_meta[op_name]

            # ランダムマグニチュードが有効な場合、マグニチュードをランダムに選択
            if self.cfg.augment.ra.random_magnitude:
                self.magnitude = torch.randint(len(mag_range), (1,), dtype=torch.long)

            # 選択されたマグニチュードを取得
            selected_mag = float(mag_range[self.magnitude].item()) if len(mag_range) > 1 else 0.0
            
            # 符号付きの場合、50%の確率で符号を反転
            if signed and torch.randint(2, (1,)):
                selected_mag *= -1.0

            # 重み付け方法に応じて拡張操作を適用
            if self.weight_type == "affinity":
                if random.random() < self.iden_rate:
                    op_name = list(self.op_meta.keys())[0]  # Identity
                else:
                    augmented_sample = _apply_op({'image': img, 'label': label}, op_name, selected_mag, interpolation=self.interpolation, fill=fill)
                    img, label = augmented_sample['image'], augmented_sample['label']
            else:
                augmented_sample = _apply_op({'image': img, 'label': label}, op_name, selected_mag, interpolation=self.interpolation, fill=fill)
                img, label = augmented_sample['image'], augmented_sample['label']

            self.count_dict[op_name] += 1

        # 一定間隔で選択された拡張操作の履歴を保存
        if self.count % self.cfg.learn.batch_size == 0:
            if self.cfg.save.selected:
                self.save_history()

        # 拡張された画像とラベルを返す
        return {'image': img, 'label': label}
    
    def save_history(self):
        pass

    def __repr__(self) -> str:
        # クラスの文字列表現を返す
        return (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}, "
            f"magnitude={self.magnitude}, "
            f"num_magnitude_bins={self.num_magnitude_bins}, "
            f"interpolation={self.interpolation}, "
            f"fill={self.fill}"
            f")"
        )

    def get_weight(self, weight):
        # 重み付け方法に応じて重みを計算
        if self.weight_type == "random":
            # ランダムな場合、全ての操作に等しい重みを与える
            weight_value = np.ones(len(weight))
            weight_value = weight_value / sum(weight_value)

        elif self.weight_type == "single":
            weight[self.single] = 1.0
            weight_value = np.array(list(weight.values()))

        else:
            weight_value = self.metrics_value

            if self.cfg.augment.ra.fix_iden:
                weight_value = np.delete(weight_value, 0)
                weight_value = nn.functional.softmax(
                    torch.tensor(weight_value) / self.cfg.augment.ra.softmax_t,
                    dim=0
                    )
                weight_value = torch.cat([torch.tensor([0,0], dtype=torch.float64), weight_value])

            else:
                weight_value = nn.functional.softmax(
                    torch.tensor(weight_value) / self.cfg.augment.ra.softmax_t,
                    dim=0
                )

        # 計算された重みを各操作に割り当てる
        for i, key in enumerate(weight):
            weight[key] = weight_value[i]

        return weight
    
    def get_metrics_values(self, weight):
        file_path = (self.cfg.out_dir + "affinity.csv")

        if not os.path.exists(file_path):
            if self.cfg.augment.ra.affinity_path is None:
                raise ValueError("affinity.csv path not found...")
            else:
                file_path = self.cfg.augment.ra.affinity_path

        df = pd.read_csv(file_path)
        for key in weight:
            weight[key] = df.iloc[-1][key]

        values = np.array(list(weight.values()))

        return values