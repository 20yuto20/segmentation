import os
import shutil

def copy_files(source_dir, target_dir, file_list, extension):
    # ターゲットディレクトリが存在しない場合は作成
    os.makedirs(target_dir, exist_ok=True)
    
    # ファイルリストを読み込む
    with open(file_list, 'r') as f:
        files = f.read().splitlines()
    
    copied_count = 0
    for file in files:
        source_file = os.path.join(source_dir, file + extension)
        target_file = os.path.join(target_dir, file + extension)
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            copied_count += 1
    
    return copied_count, len(files)

# パスの設定
voc_image_dir = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc/VOCdevkit/VOC2012/JPEGImages"
voc_label_dir = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc/VOCdevkit/VOC2012/SegmentationClass"
val_list = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc_aug/val/val.txt"
target_image_dir = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc_aug/val/image"
target_label_dir = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc_aug/val/label"

# 画像のコピー
copied_images, total_images = copy_files(voc_image_dir, target_image_dir, val_list, ".jpg")
print(f"Copied {copied_images} out of {total_images} images.")
if copied_images == total_images:
    print("All images in val.txt have been successfully copied.")
else:
    print(f"Warning: {total_images - copied_images} images were not found or couldn't be copied.")

# ラベル（セグメンテーション）画像のコピー
copied_labels, total_labels = copy_files(voc_label_dir, target_label_dir, val_list, ".png")
print(f"Copied {copied_labels} out of {total_labels} label images.")
if copied_labels == total_labels:
    print("All label images in val.txt have been successfully copied.")
else:
    print(f"Warning: {total_labels - copied_labels} label images were not found or couldn't be copied.")