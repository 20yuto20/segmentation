import argparse
import datetime
import glob
import os
import pickle
import random
import subprocess
from multiprocessing import Pool

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='aa')
parser.add_argument('--dir', type=str, help='dataset dir')
parser.add_argument('--dataset', type=str, help='cifar10 or cifar100')
args = parser.parse_args()

def download_and_extract_cifar10():
    download_command = "wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    subprocess.run([download_command], shell=True)

    extract_command = "tar -xf cifar-10-python.tar.gz"
    subprocess.run([extract_command], shell=True)

def download_and_extract_cifar100():
    download_command = "wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    subprocess.run([download_command], shell=True)

    extract_command = "tar -xf cifar-100-python.tar.gz"
    subprocess.run([extract_command], shell=True)


def save_image_parallel_cifar10(pool_list):
    label, data, name, output_dir_path, phase = pool_list

    out_dir = output_dir_path + f"/cifar10/{phase}/{label}"
    os.makedirs(out_dir, exist_ok=True)

    img = data.reshape(3, 32, 32)
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray(img)
    img.save(f"{out_dir}/{name.decode('utf-8')}")

def save_image_parallel_cifar100(pool_list):
    label, data, name, output_dir_path, phase = pool_list

    out_dir = output_dir_path + f"/cifar100/{phase}/{label}"
    os.makedirs(out_dir, exist_ok=True)

    img = data.reshape(3, 32, 32)
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray(img)
    img.save(f"{out_dir}/{name.decode('utf-8')}")

def save_cifar10_images(output_dir_path, phase):
    # phase: train/test
    if phase == "train":
        file_path_list = [f"{args.dir}/cifar-10-batches-py/data_batch_{idx}" for idx in range(1, 6)]
    elif phase == "test":
        file_path_list = [f"{args.dir}/cifar-10-batches-py/test_batch"]
    else:
        assert "no such phase!"

    pool_list = []
    for path2file in file_path_list:
        with open(path2file, "rb") as f:
            dict_data = pickle.load(f, encoding="bytes")
            for label, data, name in zip(dict_data[b"labels"], dict_data[b"data"], dict_data[b"filenames"]):
                pool_list.append([label, data, name, output_dir_path, phase])
    p = Pool(8)
    p.map(save_image_parallel_cifar10, pool_list)

def save_cifar100_images(output_dir_path, phase):
    # phase: train/test
    if phase == "train":
        file_path_list = [f"{args.dir}/cifar-100-python/train"]
    elif phase == "test":
        file_path_list = [f"{args.dir}/cifar-100-python/test"]
    else:
        assert "no such phase!"

    pool_list = []

    for path2file in file_path_list:
        with open(path2file, "rb") as f:
            dict_data = pickle.load(f, encoding="bytes")
            for key in dict_data:
                print(key)
            for label, data, name in zip(dict_data[b"fine_labels"], dict_data[b"data"], dict_data[b"filenames"]):
                pool_list.append([label, data, name, output_dir_path, phase])
    p = Pool(8)
    p.map(save_image_parallel_cifar100, pool_list)


def split_train_val_parallel(pool_list):
    path2img, class_id, out_dir = pool_list
    command = f"mv {path2img} {out_dir}/{class_id}/"
    subprocess.run([command], shell=True)


def make_val(output_dir_path, name, train_size, val_size):
    print(f"split Train:Val = {train_size}:{val_size}")
    if name == "cifar10":
        n_class = 10
    elif name == "cifar100":
        n_class = 100
    class_train_size = int(np.floor(train_size / n_class))
    class_val_size = int(np.floor(val_size / n_class))
    out_dir = output_dir_path + f"{name}/val"
    os.makedirs(out_dir, exist_ok=True)
    print(len(["train size for a class" for _ in range(class_train_size)]))
    print(len(["val size for a class" for _ in range(class_val_size)]))
    phase_list = ["train" for _ in range(class_train_size)] + ["val" for _ in range(class_val_size)]
    random.shuffle(phase_list)

    pool_list = []
    for class_id in range(n_class):
        os.makedirs(f"{out_dir}/{class_id}/", exist_ok=True)
        img_list = sorted(glob.glob(output_dir_path + f"{name}/train/{class_id}/*.png"))
        count = 0
        for path2img, phase in zip(img_list, phase_list):
            if phase == "val":
                count += 1
                pool_list.append([path2img, class_id, out_dir])
        print(f"for {class_id}, n_val: {count}")
    p = Pool(8)
    p.map(split_train_val_parallel, pool_list)


def save_log(output_dir_path, r_seed, name, train_size, val_size):
    out_path = output_dir_path + f"{name}/log.txt"
    with open(out_path, "w") as f:
        f.write(f"Create: {datetime.datetime.now()}\n")
        f.write(f"Train:val={train_size}:{val_size}, randomseed={r_seed}")


def create_CIFAR10_dataset(output_dir_path, r_seed, val_size):
    train_size = 50000 - val_size
    np.random.seed(r_seed)
    random.seed(r_seed)
    #download_and_extract_cifar10()

    save_cifar10_images(output_dir_path, phase="train")
    save_cifar10_images(output_dir_path, phase="test")
    make_val(output_dir_path, "cifar10" , train_size, val_size)

    save_log(output_dir_path, r_seed, "cifar10", train_size, val_size)


def create_CIFAR100_dataset(output_dir_path, r_seed, val_size):
    train_size = 50000 - val_size
    np.random.seed(r_seed)
    random.seed(r_seed)
    #download_and_extract_cifar100()


    save_cifar100_images(output_dir_path, phase="train")
    save_cifar100_images(output_dir_path, phase="test")
    make_val(output_dir_path, "cifar100", train_size, val_size)

    save_log(output_dir_path, r_seed, "cifar100", train_size, val_size)

output_dir_path = f"{args.dir}/"
print(f"Create {args.dataset} dataset at {args.dir}")
if args.dataset == "cifar10":
    create_CIFAR10_dataset(output_dir_path, 1, 10000)
elif args.dataset == "cifar100":
    create_CIFAR100_dataset(output_dir_path, 1, 10000)
