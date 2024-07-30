import os
import shutil
from torchvision.datasets import VOCSegmentation
from tqdm import tqdm

def download_voc(root, year, image_set='train', download=True):
    try:
        dataset = VOCSegmentation(root=root, year=year, image_set=image_set, download=download)
        print(f"Successfully downloaded VOCSegmentation {year} {image_set}")
        return dataset
    except Exception as e:
        print(f"Failed to download VOCSegmentation {year} {image_set}: {str(e)}")
        return None

def download_sbd(root):
    sbd_dir = os.path.join(root, "SBD")
    if os.path.exists(sbd_dir) and os.path.isdir(sbd_dir):
        print("SBD dataset found. Skipping download.")
        return True
    else:
        print("SBD dataset not found. Please download it manually and place it in the correct directory.")
        return False

def copy_with_progress(src, dst, description):
    total_size = os.path.getsize(src)
    with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
        with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
            while True:
                buffer = fsrc.read(8192)
                if not buffer:
                    break
                fdst.write(buffer)
                pbar.update(len(buffer))

# Set the root directory for dataset download
root_dir = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/VocSbd"

# Download PASCAL VOC 2012 (train and val)
voc_2012_dir = os.path.join(root_dir, "VOC2012")
voc_2012_train = download_voc(voc_2012_dir, "2012", "train")
voc_2012_val = download_voc(voc_2012_dir, "2012", "val")

# Download PASCAL VOC 2007 (test)
voc_2007_dir = os.path.join(root_dir, "VOC2007")
voc_2007_test = download_voc(voc_2007_dir, "2007", "test")

# Download SBD (or check if it exists)
sbd_dir = os.path.join(root_dir, "SBD")
sbd_dataset_exists = download_sbd(root_dir)

if voc_2012_train and voc_2012_val and voc_2007_test and sbd_dataset_exists:
    print("All datasets are available.")
    
    # Merge SBD into VOC2012 directory structure
    print("Merging SBD into VOC2012 directory structure...")
    
    # Copy SBD images
    sbd_img_dir = os.path.join(sbd_dir, "img")
    voc_img_dir = os.path.join(voc_2012_dir, "JPEGImages")
    os.makedirs(voc_img_dir, exist_ok=True)
    for img in tqdm(os.listdir(sbd_img_dir), desc="Copying SBD images"):
        src = os.path.join(sbd_img_dir, img)
        dst = os.path.join(voc_img_dir, img)
        if not os.path.exists(dst):
            copy_with_progress(src, dst, f"Copying {img}")

    # Copy SBD annotations
    sbd_cls_dir = os.path.join(sbd_dir, "cls")
    voc_seg_dir = os.path.join(voc_2012_dir, "SegmentationClass")
    os.makedirs(voc_seg_dir, exist_ok=True)
    for ann in tqdm(os.listdir(sbd_cls_dir), desc="Copying SBD annotations"):
        src = os.path.join(sbd_cls_dir, ann)
        dst = os.path.join(voc_seg_dir, ann)
        if not os.path.exists(dst):
            copy_with_progress(src, dst, f"Copying {ann}")

    print("Datasets merged successfully.")
else:
    print("Failed to download or locate one or more datasets.")