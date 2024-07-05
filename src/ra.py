import random
import torch
import torchvision.transforms.functional as F
import numpy as np
from torch import Tensor, nn
from typing import List, Dict, Optional, Tuple

from aug_meta import DefineAugmentSpace, _apply_op

class RandAugmentSegmentation(torch.nn.module):
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

        # self.single = cfg.augment.ra.single

        self.space = DefineAugmentSpace()
        
        if self.cfg.augment.ra.space == "ra":
            self.op_meta = self.space._ra_augmentation_space(self.num_magnitude_bins, (32, 32))
        elif self.cfg.augment.ra.space == "jda":
            self.op_meta = self.space._jda_augmentation_space(self.num_magnitude_bins, (32, 32))

        self.weight = {key: 0 for key in self.op_meta.keys()}
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

        # 指定された回数だけ拡張操作を適用
        for _ in range(self.num_ops):
            # 重みに基づいて拡張操作を選択
            op_index = torch.multinomial(torch.tensor(list(self.weight.values())), 1, replacement=True).item()
            op_name = list(self.op_meta.keys())[op_index]
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
            if self.weight_type == "random":
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

    def get_weight(self, weight):
        # 重み付け方法に応じて重みを計算
        if self.weight_type == "random":
            # ランダムな場合、全ての操作に等しい重みを与える
            weight_value = np.ones(len(weight))
            weight_value = weight_value / sum(weight_value)
        else:
            raise ValueError(f"Invalid weight type: {self.weight_type}")

        # 計算された重みを各操作に割り当てる
        for i, key in enumerate(weight):
            weight[key] = weight_value[i]

        return weight

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