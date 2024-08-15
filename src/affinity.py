import torch
import numpy as np
import pandas as pd
import copy

from torch.utils.data import DataLoader

from dataloader import get_composed_transform
from load_dataset.voc import VOCDataset, datapath_list
from ra import RandAugmentSegmentation
from utils.suggest import suggest_network
from evalator import Evaluator

class Affinity():
    def __init__(self, cfg, device):
        self.cfg = copy.deepcopy(cfg)
        self.device = device

        # get ra space
        self.cfg.augment.ra.weight = "random"
        ra = RandAugmentSegmentation(cfg=self.cfg, num_ops=self.cfg.augment.ra.num_op, magnitude=self.cfg.augment.ra.magnitude)
        self.ra_space_dict = ra.weight

        # get dataloader for each transform
        self.dataloaders = {}
        self.cfg.augment.name = ["ra"]
        self.cfg.augment.ra.weight = "single"
        dataset_path = f"{self.cfg.default.dataset_dir}"
        path_val = dataset_path + "val/"
        _, _, val_img_list, val_anno_list, _, _ = datapath_list(
            path_train=dataset_path + "train_aug/",
            path_val=path_val,
            path_test=dataset_path + "test_2007/"
        )
        for key in self.ra_space_dict:
            self.cfg.augment.ra.single = key
            aug_trans = get_composed_transform(self.cfg, "train")
            aug_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=aug_trans, img_size=self.cfg.dataset.resized_size)
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
        orig_val_miou = self.get_orig_val_miou(model)

        result_dict = {key: 0 for key in self.ra_space_dict.keys()}
        for key in result_dict:
            aug_val_loader = self.dataloaders[key]

            aug_val_miou = self.calc_val_miou(model, aug_val_loader)
            affinity_value = aug_val_miou / orig_val_miou
            result_dict[key] = affinity_value
            print(
                "Calc Affinity ..."
                + f"{key} \t"
                + f"Val mIoU: {aug_val_miou:.6f} \t"
                + f"Affinity: {affinity_value:.6f} \t"
            )

        df = pd.DataFrame(
            np.array([list(result_dict.values())]),
            columns=list(result_dict.keys()),
            )
        total_df = pd.concat([total_df, df])
        print(df)

        return total_df

    def calc_val_miou(self, model, val_loader):
        model.eval()
        evaluator = Evaluator(self.cfg.dataset.n_class)
        with torch.no_grad():
            for sample_batched in val_loader:
                image, target = sample_batched['image'].to(self.device), sample_batched['label'].to(self.device)
                output = model(image)
                pred = output.argmax(dim=1)
                evaluator.add_batch(pred.cpu().numpy(), target.cpu().numpy())

        mIoU = evaluator.Mean_Intersection_over_Union()
        return mIoU

    def get_orig_val_miou(self, model):
        dataset_path = f"{self.cfg.default.dataset_dir}"
        path_val = dataset_path + "val/"
        _, _, val_img_list, val_anno_list, _, _ = datapath_list(
            path_train=dataset_path + "train_aug/",
            path_val=path_val,
            path_test=dataset_path + "test_2007/"
        )
        orig_trans = get_composed_transform(self.cfg, "test")
        orig_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=orig_trans, img_size=self.cfg.dataset.resized_size)
        orig_loader = DataLoader(
            orig_dataset,
            batch_size=self.cfg.learn.batch_size,
            num_workers=self.cfg.default.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
            )
        orig_val_miou = self.calc_val_miou(model, orig_loader)

        return orig_val_miou

def get_affinity_init(cfg, device):
    model = suggest_network(cfg)
    model.load_state_dict(torch.load(cfg.augment.ra.aff_model))
    model.to(device)
    affinity_path = (cfg.out_dir + "affinity.csv")
    aff = Affinity(cfg, device)
    affinity_df = pd.DataFrame()
    affinity_df = aff.get_all_affinity(model, affinity_df)
    affinity_df.to_csv(affinity_path, index=False)