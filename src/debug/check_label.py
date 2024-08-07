from PIL import Image
import numpy as np

# サンプル画像のパス
sample_path = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc_aug/train_aug/label/2008_005869.png"

with Image.open(sample_path) as img:
    label_array = np.array(img)
    print(f"Shape: {label_array.shape}")
    print(f"Data type: {label_array.dtype}")
    print(f"Unique values: {np.unique(label_array)}")
    print(f"Min value: {np.min(label_array)}")
    print(f"Max value: {np.max(label_array)}")