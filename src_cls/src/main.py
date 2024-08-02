import time
import os
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import sys

import torch

from dataloader import get_dataloader
from train_val import train, val, test
from set_cfg import setup_config, add_config
from randaugment import reset_cfg
# from affinity import Affinity, get_affinity_init
from utils.suggest import (
    setup_device,
    fixed_r_seed,
    suggest_network,
    suggest_optimizer,
    suggest_scheduler,
    suggest_loss_func,
)
from utils.common import (
    plot_log, 
    plot_selected, 
    get_time,
    copy_from_sge,
    save_learner,
    lr_step
)

print("giselle")

def main(cfg):
    # if single-pass, set cfg for init phase
    if cfg.augment.name[0] == "single" or cfg.augment.name[0] == "w_ra":
        cfg = reset_cfg(cfg, init=True)
    # set num_worker
    if cfg.default.num_workers is None:
        cfg.default.num_workers=int(os.cpu_count() / cfg.default.num_excute) 
        print(f"CPU cores ..{os.cpu_count()}, num workers ... {cfg.default.num_workers}")

    device = setup_device(cfg)
    fixed_r_seed(cfg)

    # if cfg.augment.ra.aff_calc:
    #     get_affinity_init(cfg, device)

    model = suggest_network(cfg)
    model.to(device)

    train_loader, val_loader, test_loader = get_dataloader(cfg)
    optimizer = suggest_optimizer(cfg, model)
    scheduler = suggest_scheduler(cfg, optimizer)
    loss_func = suggest_loss_func()

    # # if need affinity calc, prepare csv path and instance 
    # if cfg.save.affinity or cfg.save.affinity_all:
    #     affinity_path = (cfg.out_dir + "affinity.csv")
    #     aff = Affinity(cfg, device)
        
    print(OmegaConf.to_yaml(cfg))
    start = time.time()
    best_acc = 1e-8
    save_file_path = cfg.out_dir + "output.csv"
    all_training_result = []
    # affinity_df = pd.DataFrame()
    for epoch in range(1, cfg.learn.n_epoch + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, loss_func)
        val_loss, val_acc, val_mAP = val(model, device, val_loader, loss_func)

        all_training_result.append([
            train_loss.cpu().item() if isinstance(train_loss, torch.Tensor) else train_loss,
            train_acc.cpu().item() if isinstance(train_acc, torch.Tensor) else train_acc,
            val_loss.cpu().item() if isinstance(val_loss, torch.Tensor) else val_loss,
            val_acc.cpu().item() if isinstance(val_acc, torch.Tensor) else val_acc,
            val_mAP.cpu().item() if isinstance(val_mAP, torch.Tensor) else val_mAP
        ])
        interval = time.time() - start
        interval = get_time(interval)
        print(f"Lr: {optimizer.param_groups[0]['lr']} , time: {interval['time']}")
        print(
            f"Epoch: [{epoch:03}/{cfg.learn.n_epoch:03}] \t"
            + f"train loss: {train_loss:.6f} \t"
            + f"train acc: {train_acc:.6f} \t"
            + f"val loss: {val_loss:.6f} \t"
            + f"val acc: {val_acc:.6f} \t"
        )
        sys.stdout.flush()

        # # if init phase end
        # if cfg.augment.ra.init_epoch is not None and epoch == cfg.augment.ra.init_epoch:
        #     # for single-pass calc Affinity using the model at that epoch
        #     affinity_df = aff.get_all_affinity(model, affinity_df)
        #     affinity_df.to_csv(affinity_path, index=False)
        #     # start main phase, change dataloader for RA
        #     cfg = reset_cfg(cfg, init=False)
        #     print("Switch to new data loader ...")
        #     del train_loader
        #     train_loader, _, _ = get_dataloader(cfg)

        # # if affinity at each epoch calc
        # elif cfg.save.affinity_all and epoch % 2 == 0:
        #     affinity_df = aff.calculate_affinity(model, val_acc, epoch, affinity_df)
        #     affinity_df.to_csv(affinity_path, index=False)
        
        # save best weight
        # TODO: multilabelではどの評価指標をもとに重み付けをするか
        if best_acc < val_acc:
            best_acc = val_acc
            save_learner(cfg, model, device, True)
        # save latest epoch weight
        save_learner(cfg, model, device, False)

        scheduler.step()

    all_training_result = pd.DataFrame(
        all_training_result,
        columns=["train_loss", "train_acc", "val_loss", "val_acc", "val_mAP"]
    )
    
    print("all_training_result type:", type(all_training_result))
    print("all_training_result first row:", all_training_result.iloc[0])

    interval = time.time() - start
    interval = get_time(interval)

    test_loss, test_acc, test_mAP = test(model, device, test_loader, loss_func, cfg)

    # GPUテンソルをCPUに移動し、Pythonのネイティブ型に変換
    test_loss = test_loss.cpu().item() if isinstance(test_loss, torch.Tensor) else test_loss
    test_acc = test_acc.cpu().item() if isinstance(test_acc, torch.Tensor) else test_acc
    test_mAP = test_mAP.cpu().item() if isinstance(test_mAP, torch.Tensor) else test_mAP

    print(
        f"time: {interval['time']} \t"
        +f"test loss: {test_loss:.6f} \t"
        +f"test acc: {test_acc:.6f} \t"
        +f"test mAP: {test_mAP:.6f} \t"
    )

    # DataFrameに新しい行を追加
    new_row = pd.DataFrame({
        "train_loss": [np.nan],
        "train_acc": [np.nan],
        "val_loss": [np.nan],
        "val_acc": [np.nan],
        "val_mAP": [np.nan],
        "test_loss": [test_loss],
        "test_acc": [test_acc],
        "test_mAP": [test_mAP]
    })
    all_training_result = pd.concat([all_training_result, new_row], ignore_index=True)
    all_training_result.to_csv(save_file_path, index=False)
    
    add_config(cfg, {"test_acc" : test_acc})
    add_config(cfg, interval)
    plot_log(cfg, all_training_result)

    # if cfg.save.affinity:
    #     affinity_df = aff.calculate_affinity(model, val_acc, epoch, affinity_df)
    #     affinity_df.to_csv(affinity_path, index=False)

    # if abci move files from SGE_LOCALDIR
    if cfg.default.env == "abci":
        print("Copy csv data from SGE_LOCALDIR")
        copy_from_sge(cfg, "selected_method")
    if cfg.save.selected and os.path.exists(cfg.out_dir + f"selected_method_{cfg.augment.ra.weight}.csv"):
        plot_selected(cfg)

if __name__ == "__main__":
    cfg = setup_config()
    main(cfg)