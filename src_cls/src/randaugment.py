from typing import List, Optional
import numpy as np
import random
import pandas as pd
from omegaconf import OmegaConf
import os
from pathlib import Path

import torch
from torch import Tensor, nn
import torchvision.transforms.functional as F

from aug_meta import DefineAugmentSpace, _apply_op
from set_cfg import override_original_config


# single w_ra


# for single-pass method
def reset_cfg(cfg, init: bool):
    # set for init phase
    if init:
        if cfg.augment.ra.init_epoch is None:
            raise ValueError("error.. set num of init phase epoch")
        print(f"Set cfg for init phase, num of init phase epoch ...{cfg.augment.ra.init_epoch}")

        if cfg.augment.name[0] == "w_ra":
            cfg.augment.ra.warmup_ra = True
            print("Apply warmup RA")

        cfg.augment.name=["nan"]
        cfg.save.affinity=True

    # set for main phase
    # start RA during training
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




class RandAugment(torch.nn.Module):
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

        # weightを取得するためにRA spaceを取得，画像サイズは何でも良い
        if self.cfg.augment.ra.space == "ra":
            op_meta = self.space._ra_augmentation_space(self.num_magnitude_bins, (32, 32))
        elif self.cfg.augment.ra.space == "jda":
            op_meta = self.space._jda_augmentation_space(self.num_magnitude_bins, (32, 32))

        self.weight = {key: 0 for key in op_meta.keys()}
        if "ra" in self.cfg.augment.name:
            if self.weight_type == "affinity":
                self.metrics_value = self.get_metrics_value(self.weight)

            self.weight = self.get_weight(self.weight)
            
            # print("Weight values ...")
            # print(self.weight)
            
        # if fix Identity rate
        self.iden_rate = float(1 / len(op_meta))

        # count num of selected method
        self.count = 0
        self.count_dict = {key: 0 for key in op_meta.keys()}



    def forward(self, img: Tensor) -> Tensor:
        self.count += 1
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        if self.cfg.augment.ra.space == "ra":
            op_meta = self.space._ra_augmentation_space(self.num_magnitude_bins, (height, width))
        elif self.cfg.augment.ra.space == "jda":
            op_meta = self.space._jda_augmentation_space(self.num_magnitude_bins, (height, width))
     
        # softmax_t is random smpled for each image
        if self.cfg.augment.ra.softmax_t == "random":
            self.cfg.augment.ra.softmax_t = np.random.rand()
            self.weight = self.get_weight(self.weight)

        # apply transform num_ops times
        for _ in range(self.num_ops):
            # sample op according to weight value
            op_index = torch.multinomial(torch.tensor(list(self.weight.values())), 1, replacement=True).item()
            op_name = list(op_meta.keys())[op_index]
            mag_range, signed = op_meta[op_name]

            if self.cfg.augment.ra.random_magnitude:
                self.magnitude = torch.randint(len(mag_range), (1,), dtype=torch.long)

            ## self.magnitudeはレベル, magnitudeは手法ごとのハイパラ
            # transformのhyper-paraにrangがあるならlevelを指定，なかったら0.0を返す
            selected_mag = (float(mag_range[self.magnitude].item()) if len(mag_range) > 1 else 0.0)
            #magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            
            # if hyper-para can invert, invert at 50% prob
            if signed and torch.randint(2, (1,)):
                selected_mag *= -1.0

            # Affinityで重み付けしてapply probを決めるが，
            if self.weight_type == "affinity":
                # for a certain prob -> Identity
                if random.random() < self.iden_rate:
                    op_name = list(op_meta.keys())[0]
                # else, apply transform
                else:
                    img = _apply_op(img, op_name, selected_mag, interpolation=self.interpolation, fill=fill)
            else:
                img = _apply_op(img, op_name, selected_mag, interpolation=self.interpolation, fill=fill)

            # count selected transform
            self.count_dict[op_name] += 1

        # たまに保存
        if self.count % self.cfg.learn.batch_size == 0:
            if self.cfg.save.selected:
                self.save_history()

        return img
    
    def save_history(self):
        # for abci, save at SGE dir (temp)
        if self.cfg.default.env=="abci":
            sge_dir = str(Path(self.cfg.default.dataset_dir).parent)
            file_path = os.path.join(sge_dir, f"selected_method_{self.weight_type}.csv")
        else:
            file_path = self.cfg.out_dir + f"selected_method_{self.weight_type}.csv"

        df = pd.DataFrame([list(self.count_dict.values())], columns= list(self.count_dict.keys()))  

        for key in self.count_dict:
            self.count_dict[key] = 0
        
        # 既存csvに書き足す
        if os.path.exists(file_path) and self.count != 128:
            with open(file_path, 'a') as f:
                df.to_csv(f, header=False, index=False)
        # 1番初めはcsvを作成
        else:
            df.to_csv(file_path, index = False)
        

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s
    

    def get_weight(self, weight):
        # original RA, select at random
        if self.weight_type == "random":
            weight_value = np.ones(len(weight))
            weight_value = weight_value / sum(weight_value)

        # selct only one method
        elif self.weight_type == "single":
            weight[self.single] = 1.0
            weight_value = np.array(list(weight.values()))

        # proposed method, weight for selection prob
        else:
            weight_value = self.metrics_value
            # not weight for iden, iden prob is fixed
            if self.cfg.augment.ra.fix_iden:
                weight_value = np.delete(weight_value, 0)
                weight_value = nn.functional.softmax(
                    torch.tensor(weight_value) / self.cfg.augment.ra.softmax_t,
                    dim=0
                    )
                weight_value = torch.cat([torch.tensor([0.0], dtype=torch.float64), weight_value])
                
            else:
                weight_value = nn.functional.softmax(
                    torch.tensor(weight_value) / self.cfg.augment.ra.softmax_t,
                    dim=0
                    )

        for i, key in enumerate(weight):
            weight[key] = weight_value[i]

        return weight
   
    # get affinity value from csv
    def get_metrics_value(self, weight):
        file_path = (self.cfg.out_dir + "affinity.csv")

        # if not exist affinity.csv in same dir
        if not os.path.exists(file_path):
            # if affinity.csv path is not directed
            if self.cfg.augment.ra.affinity_path is None:
                raise ValueError("affinity.csv path not found ...")
            else:
                file_path = self.cfg.augment.ra.affinity_path

        # load affinity value from csv
        df = pd.read_csv(file_path)
        for key in weight:
            weight[key] = df.iloc[-1][key]

        values = np.array(list(weight.values()))

        return values
    
    
