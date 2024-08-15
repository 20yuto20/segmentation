import time
import os
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import sys

import torch

from dataloader import get_dataloader
from train_val import train, val, test
from evalator import Evaluator
from set_cfg import setup_config, add_config
from ra import reset_cfg
from affinity import Affinity, get_affinity_init
from utils.common import (
    setup_device,
    fixed_r_seed,
    plot_log, 
    plot_selected, 
    get_time,
    copy_from_sge,
    save_learner,
    lr_step
)
from utils.suggest import (
    suggest_network,
    suggest_optimizer,
    suggest_scheduler,
    suggest_loss_func,
)

def main(cfg):
    if cfg.augment.name[0] == "single" or cfg.augment.name[0] == "w_ra":
        cfg = reset_cfg(cfg, init=True)
    if cfg.default.num_workers is None:
        cfg.default.num_workers=int(os.cpu_count() / cfg.default.num_excute) 
        print(f"CPU cores ..{os.cpu_count()}, num workers ... {cfg.default.num_workers}")

    device = setup_device(cfg)
    fixed_r_seed(cfg)

    if cfg.augment.ra.aff_calc:
        get_affinity_init(cfg, device)

    model = suggest_network(cfg)
    model.to(device)

    train_loader, val_loader, test_loader = get_dataloader(cfg)
    optimizer = suggest_optimizer(cfg, model)
    scheduler = suggest_scheduler(cfg, optimizer)
    loss_func = suggest_loss_func(cfg)

    if cfg.save.affinity or cfg.save.affinity_all:
        affinity_path = (cfg.out_dir + "affinity.csv")
        aff = Affinity(cfg, device)
        
    print(OmegaConf.to_yaml(cfg))
    start = time.time()
    best_miou = 0.0
    save_file_path = cfg.out_dir + "output.csv"
    all_training_result = []
    affinity_df = pd.DataFrame()
    evaluator = Evaluator(cfg.dataset.n_class)  # 追加
    for epoch in range(1, cfg.learn.n_epoch + 1):
        train_loss, train_miou, train_acc = train(cfg, device, model, train_loader, optimizer, loss_func, evaluator, epoch)
        val_loss, val_miou, val_acc = val(cfg, device, model, val_loader, loss_func, evaluator, epoch)


        all_training_result.append([train_loss, train_miou, train_acc, val_loss, val_miou, val_acc])
        interval = time.time() - start
        interval = get_time(interval)
        print(f"Lr: {optimizer.param_groups[0]['lr']} , time: {interval['time']}")
        print(
            f"Epoch: [{epoch:03}/{cfg.learn.n_epoch:03}] \t"
            + f"train loss: {train_loss:.6f} \t"
            + f"train mIoU: {train_miou:.6f} \t"
            + f"train acc: {train_acc:.6f} \t"
            + f"val loss: {val_loss:.6f} \t"
            + f"val mIoU: {val_miou:.6f} \t"
            + f"val acc: {val_acc:.6f} \t"
        )
        sys.stdout.flush()

        if cfg.augment.ra.init_epoch is not None and epoch == cfg.augment.ra.init_epoch:
            affinity_df = aff.get_all_affinity(model, affinity_df)
            affinity_df.to_csv(affinity_path, index=False)
            cfg = reset_cfg(cfg, init=False)
            print("Switch to new data loader ...")
            del train_loader
            train_loader, _, _ = get_dataloader(cfg)

        elif cfg.save.affinity_all and epoch % cfg.save.interval == 0:
            affinity_df = aff.get_all_affinity(model, affinity_df)
            affinity_df.to_csv(affinity_path, index=False)
        
        if best_miou < val_miou:
            best_miou = val_miou
            save_learner(cfg, model, device, True)
        save_learner(cfg, model, device, False)

        scheduler.step()

    all_training_result = pd.DataFrame(
        np.array(all_training_result),
        columns=["train_loss", "train_miou", "train_acc", "val_loss", "val_miou", "val_acc"],
    )
    interval = time.time() - start
    interval = get_time(interval)

    test_miou, test_acc = test(cfg, device, model, test_loader, loss_func)
    print(
        f"time: {interval['time']} \t"
        +f"test mIoU: {test_miou:.6f} \t"
        +f"test acc: {test_acc:.6f} \t"
        )

    all_training_result.loc["test_miou"] = test_miou
    all_training_result.loc["test_acc"] = test_acc
    all_training_result.to_csv(save_file_path, index=False)
    
    add_config(cfg, {"test_miou" : test_miou, "test_acc" : test_acc})
    add_config(cfg, interval)
    plot_log(cfg, all_training_result)

    if cfg.default.env == "abci":
        print("Copy csv data from SGE_LOCALDIR")
        copy_from_sge(cfg, "selected_method")
    if cfg.save.selected and os.path.exists(cfg.out_dir + f"selected_method_{cfg.augment.ra.weight}.csv"):
        plot_selected(cfg)

if __name__ == "__main__":
    cfg = setup_config()
    main(cfg)