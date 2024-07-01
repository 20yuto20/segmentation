import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path 
import shutil
import random

import torch


# usible cudaの取得
# return : device
# -> model.to(device), data.to(devide) で使用
def setup_device(cfg):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.default.device_id}")
        
        if not cfg.default.deterministic:
            torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    print("CUDA is available:", torch.cuda.is_available())
    print(f"using device: {device}")
    return device


# 全体のseed値の固定
# pytorchやnumpy変数など全てのseed値が固定される
# プログラムの初めで呼び出してseedを固定する必要
def fixed_r_seed(cfg):
    random.seed(cfg.default.seed)
    np.random.seed(cfg.default.seed)
    torch.manual_seed(cfg.default.seed)
    torch.cuda.manual_seed(cfg.default.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


# start = time.time()
# end = time.time()
# interval = start - end
# intervalを入れると秒を時間，分，秒に直して辞書型で返す
def get_time(interval):
    time = {"time" : "{}h {}m {}s".format(
            int(interval / 3600), 
            int((interval % 3600) / 60), 
            int((interval % 3600) % 60))}
    return time


# plot loss and acc curve
def plot_log(cfg, data):
    # epochs = np.arange(1, len(data) + 1 - 2)
    epochs = np.arange(1, len(data) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(16, 9), dpi = 80)
    
    # ax[0].plot(epochs, data["train_loss"][:-2], label='Train', alpha=0.8, linewidth=5)
    ax[0].plot(epochs, data["train_loss"], label='Train', alpha=0.8, linewidth=5)
    ax[0].set_title('Train', fontsize=30)
    ax[0].set_xlabel('Epochs', fontsize=25)
    ax[0].set_ylabel('Loss', fontsize=25)
    ax[0].legend(bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=0.2, fontsize=30, ncol=1)
    ax[0].tick_params(labelsize=25)
    ax[0].grid()

    ax[1].plot(epochs, data["val_mIoU"], label='mIoU', alpha=0.8, linewidth=5)
    ax[1].plot(epochs, data["val_acc"], label='acc', alpha=0.8, linewidth=5)
    ax[1].set_title('Validation', fontsize=30)
    ax[1].set_xlabel('Epochs', fontsize=25)
    ax[1].set_ylabel('Accuracy', fontsize=25)
    ax[1].legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=0.2, fontsize=30, ncol=1)
    ax[1].tick_params(labelsize=25)
    ax[1].grid()
    
    fig.suptitle(f"{cfg.out_dir}")
    plt.tight_layout()
    plt.savefig(cfg.out_dir + "graph.png")
    plt.close()


# show sample 12 imgs
def show_img(cfg, dataloader):
    for batched in dataloader:
        images = batched["image"]
        labels = batched["label"]
        break
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for i in range(12):
        ax = axes[i // 4, i % 4]
        img = np.transpose(images[i].numpy(), (1, 2, 0))  
        ax.imshow(img)
        ax.set_title(f"Label: {labels[i]}")
        ax.axis('off')

    plt.savefig(cfg.out_dir + "img.png")
    plt.close()



# Afinity-Weighted RA使用時に使う
# plot the num of selected method (read from csv file)
def plot_selected(cfg):
    file_path = cfg.out_dir + f"selected_method_{cfg.augment.ra.weight}.csv"

    df = pd.read_csv(file_path)

    fig, ax = plt.subplots(figsize=(16, 9))  

    interval = cfg.learn.n_epoch*10
    if type(cfg.save.interval) == int:
        interval = cfg.save.interval
    for method in df.columns:
        values = df.loc[:, method]
        ax.plot(range(1, len(values)+1, interval), values.iloc[::interval], label=method, marker='o', markersize=2)

    ax.set_xlabel('Iteration', fontsize=25)
    ax.set_ylabel('Count', fontsize=25)
    ax.set_title('Selected method', fontsize=30)
    fig.suptitle(f"{cfg.out_dir}")
    ax.tick_params(labelsize=25)
    ax.legend()
    plt.grid(True)
    plt.savefig(cfg.out_dir + "selected.png")
    plt.close()


# abci使用時に使う
def copy_from_sge(cfg, target_dir_name):
    sge_dir = str(Path(cfg.default.dataset_dir).parent)
    files = [f for f in os.listdir(sge_dir) if os.path.isfile(os.path.join(sge_dir, f))]
    for file in files:
        if target_dir_name in file:
            source_path = os.path.join(sge_dir, file)
            destination_path = os.path.join(cfg.out_dir, file)
            shutil.copy2(source_path, destination_path)


# abci使用時に使う
def copy_to_sge(cfg, target_path):
    sge_dir = str(Path(cfg.default.dataset_dir).parent)
    shutil.copytree(target_path, sge_dir)


# 学習済みモデルの重みを保存
# if BEST == False:
#   学習途中のモデル重みを保存するときに使う
#   weight/latest_epochs.pth に保存される
# elif BEST == True:
#   val accが最良のモデル重みを保存するときに使う
#   weight/best.pth に保存される
def save_learner(cfg, model, device, BEST=False):
    weight_dir_path = cfg.out_dir + "weights/"
    os.makedirs(weight_dir_path, exist_ok=True)
    if BEST:
        save_file_path = weight_dir_path + "best.pth"
    else:
        save_file_path = weight_dir_path + "latest_epochs.pth"

    torch.save(
        model.to("cpu").state_dict(),
        save_file_path,
    )
    model.to(device)


# 毎エポックのモデル重みを保存する時に使用，基本的には使わない
# def save_all_learner(cfg, model, device, epoch):
#     weight_dir_path = cfg.out_dir + "weights/"
#     os.makedirs(weight_dir_path, exist_ok=True)
#     save_file_path = weight_dir_path + f"{epoch}.pth"

#     torch.save(
#         model.to("cpu").state_dict(),
#         save_file_path,
#     )
#     model.to(device)


def lr_step(cfg, scheduler, epoch):
        if cfg.optimizer.scheduler.name == "warmup":
            scheduler.step(epoch)

        else:
            scheduler.step()

