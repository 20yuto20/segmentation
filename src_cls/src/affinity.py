import numpy as np
import pandas as pd
import copy
import os

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from dataloader import get_composed_transform, VOCDatasetLoader
from randaugment import RandAugment
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
        dataset_path = self.cfg.default.dataset_dir  # This should now point directly to the VOCdevkit directory
        for key in self.ra_space_dict:
            self.cfg.augment.ra.single = key
            aug_trans = get_composed_transform(self.cfg, "train")
            aug_dataset = VOCDatasetLoader(dataset_path, '2012', 'val', self.cfg.dataset.resized_size, aug_trans)
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

    def get_all_affinity(self, model, total_df):
        orig_val_mAP = self.get_orig_val_mAP(model)

        result_dict = {key: 0 for key in self.ra_space_dict.keys()}
        for key in result_dict:
            aug_val_loader = self.dataloaders[key]

            aug_val_mAP = self.calc_val_mAP(model, aug_val_loader)
            affinity_value = aug_val_mAP / orig_val_mAP
            result_dict[key] = affinity_value
            print(
                "Calc Affinity ..."
                + f"{key} \t"
                + f"Val mAP: {aug_val_mAP:.6f} \t"
                + f"Affinity: {affinity_value:.6f} \t"
            )

        df = pd.DataFrame(
            np.array([list(result_dict.values())]),
            columns=list(result_dict.keys()),
            )
        total_df = pd.concat([total_df, df])
        print(df)

        return total_df

    def calculate_affinity(self, model, val_mAP, epoch, affinity_df):
        orig_val_mAP = val_mAP

        result_dict = {key: 0 for key in self.ra_space_dict.keys()}
        for key in result_dict:
            aug_val_loader = self.dataloaders[key]

            aug_val_mAP = self.calc_val_mAP(model, aug_val_loader)
            affinity_value = aug_val_mAP / orig_val_mAP
            result_dict[key] = affinity_value
            print(
                f"Epoch {epoch}, Calc Affinity ..."
                + f"{key} \t"
                + f"Val mAP: {aug_val_mAP:.6f} \t"
                + f"Affinity: {affinity_value:.6f} \t"
            )

        df = pd.DataFrame(
            np.array([list(result_dict.values())]),
            columns=list(result_dict.keys()),
            )
        affinity_df = pd.concat([affinity_df, df])
        print(df)

        return affinity_df

    def calc_val_mAP(self, model, val_loader):
        model.eval()
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for sample_batched in val_loader:
                data, target = sample_batched["image"].to(self.device), sample_batched["label"].to(self.device)
                output = model(data)
                all_outputs.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)

        ap_scores = []
        for i in range(all_targets.shape[1]):
            ap = average_precision_score(all_targets[:, i], all_outputs[:, i])
            ap_scores.append(ap)

        mAP = np.mean(ap_scores)
        return mAP

    def get_orig_val_mAP(self, model):
        dataset_path = self.cfg.default.dataset_dir  # This should now point directly to the VOCdevkit directory
        orig_trans = get_composed_transform(self.cfg, "test")
        orig_dataset = VOCDatasetLoader(dataset_path, '2012', 'val', self.cfg.dataset.resized_size, orig_trans)
        orig_loader = DataLoader(
            orig_dataset,
            batch_size=self.cfg.learn.batch_size,
            num_workers=self.cfg.default.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
            )
        orig_val_mAP = self.calc_val_mAP(model, orig_loader)

        return orig_val_mAP

def get_affinity_init(cfg, device):
    model = suggest_network(cfg)
    model.load_state_dict(torch.load(cfg.augment.ra.aff_model))
    model.to(device)
    affinity_path = (cfg.out_dir + "affinity.csv")
    aff = Affinity(cfg, device)
    affinity_df = pd.DataFrame()
    affinity_df = aff.get_all_affinity(model, affinity_df)
    affinity_df.to_csv(affinity_path, index=False)