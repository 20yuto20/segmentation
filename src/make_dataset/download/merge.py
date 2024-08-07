import os
import shutil
import scipy.io
import numpy as np
from PIL import Image
from tqdm import tqdm

def mat_to_png(mat_file, png_file):
    mat = scipy.io.loadmat(mat_file)
    segmentation = mat['GTcls']['Segmentation'][0][0]
    img = Image.fromarray(segmentation.astype(np.uint8))
    img.save(png_file)

root_dir = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/VocSbd"
sbd_dir = os.path.join(root_dir, "SBD")
voc_dir = os.path.join(root_dir, "VOC2012")

# Create SegmentationClassAug directory
voc_aug_dir = os.path.join(voc_dir, "SegmentationClassAug")
os.makedirs(voc_aug_dir, exist_ok=True)

# Convert and copy SBD segmentation masks
print("Converting and copying SBD segmentation masks...")
sbd_cls_dir = os.path.join(sbd_dir, "cls")
for mat_file in tqdm(os.listdir(sbd_cls_dir)):
    if mat_file.endswith('.mat'):
        mat_path = os.path.join(sbd_cls_dir, mat_file)
        png_file = os.path.splitext(mat_file)[0] + '.png'
        png_path = os.path.join(voc_aug_dir, png_file)
        mat_to_png(mat_path, png_path)

# Copy train.txt and val.txt
print("Copying train.txt and val.txt...")
for file in ['train.txt', 'val.txt']:
    src = os.path.join(sbd_dir, file)
    dst = os.path.join(voc_dir, file)
    shutil.copy2(src, dst)

print("SBD dataset preparation completed.")