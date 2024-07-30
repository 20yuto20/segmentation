import os
import shutil
from tqdm import tqdm
import random

# 基本ディレクトリの設定
base_dir = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/vocsbd"
voc_dir = os.path.join(base_dir, "VOC2012")
vocdevkit_dir = os.path.join(voc_dir, "VOCdevkit", "VOC2012")
vocsbd_dir = os.path.join(base_dir, "VOCSBD")

# 新しいディレクトリの作成
os.makedirs(os.path.join(vocsbd_dir, "JPEGImages"), exist_ok=True)
os.makedirs(os.path.join(vocsbd_dir, "SegmentationClass"), exist_ok=True)
os.makedirs(os.path.join(vocsbd_dir, "split"), exist_ok=True)

# 1. JPEGImagesの統合
print("Merging JPEGImages...")
sbd_jpg_dir = os.path.join(voc_dir, "JPEGImages")
voc_jpg_dir = os.path.join(vocdevkit_dir, "JPEGImages")
merged_jpg_dir = os.path.join(vocsbd_dir, "JPEGImages")

for jpg_dir in [sbd_jpg_dir, voc_jpg_dir]:
    for img in tqdm(os.listdir(jpg_dir)):
        if img.endswith('.jpg'):
            src = os.path.join(jpg_dir, img)
            dst = os.path.join(merged_jpg_dir, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

# 2. SegmentationClassの統合
print("Merging SegmentationClass...")
sbd_seg_dir = os.path.join(voc_dir, "SegmentationClassAug")
voc_seg_dir = os.path.join(vocdevkit_dir, "SegmentationClass")
merged_seg_dir = os.path.join(vocsbd_dir, "SegmentationClass")

for seg_dir in [sbd_seg_dir, voc_seg_dir]:
    for img in tqdm(os.listdir(seg_dir)):
        if img.endswith('.png'):
            src = os.path.join(seg_dir, img)
            dst = os.path.join(merged_seg_dir, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

# 3. split ファイルの統合
print("Merging split files...")
sbd_train = os.path.join(voc_dir, "train.txt")
sbd_val = os.path.join(voc_dir, "val.txt")
voc_train = os.path.join(vocdevkit_dir, "ImageSets", "Segmentation", "train.txt")
voc_val = os.path.join(vocdevkit_dir, "ImageSets", "Segmentation", "val.txt")

merged_train = os.path.join(vocsbd_dir, "split", "train.txt")
merged_val = os.path.join(vocsbd_dir, "split", "val.txt")

def merge_txt_files(file1, file2, output_file):
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w') as out:
        lines = set(f1.read().splitlines() + f2.read().splitlines())
        out.write('\n'.join(sorted(lines)))

merge_txt_files(sbd_train, voc_train, merged_train)
merge_txt_files(sbd_val, voc_val, merged_val)

# 4. テスト: 画像とラベルの対応確認
print("Testing image-label correspondence...")

def check_correspondence(split_file, jpg_dir, seg_dir):
    with open(split_file, 'r') as f:
        files = f.read().splitlines()
    
    mismatched = []
    for file in tqdm(files):
        jpg_path = os.path.join(jpg_dir, f"{file}.jpg")
        png_path = os.path.join(seg_dir, f"{file}.png")
        if not (os.path.exists(jpg_path) and os.path.exists(png_path)):
            mismatched.append(file)
    
    return mismatched

train_mismatched = check_correspondence(merged_train, merged_jpg_dir, merged_seg_dir)
val_mismatched = check_correspondence(merged_val, merged_jpg_dir, merged_seg_dir)

if not train_mismatched and not val_mismatched:
    print("Correlation is correct")
else:
    print("Mismatches found. Creating relation.txt...")
    relation_file = os.path.join(vocsbd_dir, "split", "relation.txt")
    with open(relation_file, 'w') as f:
        f.write("Image_Name\tJPEG_Exists\tPNG_Exists\n")
        all_files = set(os.listdir(merged_jpg_dir) + os.listdir(merged_seg_dir))
        for file in all_files:
            name = os.path.splitext(file)[0]
            jpg_exists = os.path.exists(os.path.join(merged_jpg_dir, f"{name}.jpg"))
            png_exists = os.path.exists(os.path.join(merged_seg_dir, f"{name}.png"))
            f.write(f"{name}\t{jpg_exists}\t{png_exists}\n")

print("Data integration and testing completed.")