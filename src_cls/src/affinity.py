import numpy as np
import pandas as pd
import copy

import torch
from torch.utils.data import DataLoader

from dataloader import get_composed_transform, DatasetLoader
from randaugment import RandAugment
from train_val import accuracy
from utils.suggest import suggest_network



class Affinity():
    def __init__(self, cfg, device):
        self.cfg = copy.deepcopy(cfg)
        self.device = device

        # get ra space
        self.cfg.augment.ra.weight = "random"
        ra = RandAugment(cfg=self.cfg, num_ops=self.cfg.augment.ra.num_op, magnitude=self.cfg.augment.ra.magnitude)
        self.ra_space_dict = ra.weight

        # get dataloader for each transform
        self.dataloaders = {}
        self.cfg.augment.name = ["ra"]
        self.cfg.augment.ra.weight = "single"
        dataset_path = f"{self.cfg.default.dataset_dir}"+ f"{self.cfg.dataset.name}" 
        for key in self.ra_space_dict:
            self.cfg.augment.ra.single = key
            aug_trans = get_composed_transform(self.cfg, "train")
            aug_dataset = DatasetLoader(dataset_path, "val", self.cfg.dataset.resized_size, aug_trans)
            aug_loader = DataLoader(
                aug_dataset,
                batch_size=self.cfg.learn.batch_size,
                num_workers=self.cfg.default.num_workers,
                shuffle=False,
                pin_memory=True,
                persistent_workers=True,
                drop_last=False,
                )

            self.dataloaders[key] = aug_loader


    # 空のdfに結果を格納して返すか，
    # すでにaffinityが入っっていたら追加
    def get_all_affinity(self, model, total_df):
        orig_val_acc = self.get_orig_val_acc(model)

        result_dict = {key: 0 for key in self.ra_space_dict.keys()}
        for key in result_dict:
            aug_val_loader = self.dataloaders[key]

            aug_val_acc = self.calc_val_acc(model, aug_val_loader)
            affinity_value = aug_val_acc / orig_val_acc
            result_dict[key] = affinity_value
            print(
                "Calc Affinity ..."
                + f"{key} \t"
                + f"Val Acc: {aug_val_acc:.6f} \t"
                + f"Affinity: {affinity_value:.6f} \t"
            )

        df = pd.DataFrame(
            np.array([list(result_dict.values())]),
            columns=list(result_dict.keys()),
            )
        total_df = pd.concat([total_df, df])
        print(df)

        return total_df
    

    def calc_val_acc(self, model, val_loader):
        model.eval()
        val_acc, n_val = 0, 0
        model.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(val_loader):
                if i_batch == 1:
                    break
                data, target = sample_batched["image"].to(self.device), sample_batched["label"].to(self.device)
                output = model(data)
                val_acc += accuracy(output, target)
                n_val += target.size(0)
                
        val_acc = float(val_acc) / n_val

        return val_acc
    

    def get_orig_val_acc(self, model):
        dataset_path = f"{self.cfg.default.dataset_dir}"+ f"{self.cfg.dataset.name}" 
        orig_trans = get_composed_transform(self.cfg, "test")
        orig_dataset = DatasetLoader(dataset_path, "val", self.cfg.dataset.resized_size, orig_trans)
        orig_loader = DataLoader(
            orig_dataset,
            batch_size=self.cfg.learn.batch_size,
            num_workers=self.cfg.default.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
            )
        orig_val_acc = self.calc_val_acc(model, orig_loader)

        return orig_val_acc



def get_affinity_init(cfg, device):
    model = suggest_network(cfg)
    model.load_state_dict(torch.load(cfg.augment.ra.aff_model))
    model.to(device)
    affinity_path = (cfg.out_dir + "affinity.csv")
    aff = Affinity(cfg, device)
    affinity_df = pd.DataFrame()
    affinity_df = aff.get_all_affinity(model, affinity_df)
    affinity_df.to_csv(affinity_path, index=False)