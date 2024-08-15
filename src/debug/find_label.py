import os
from PIL import Image
import numpy as np

label_dir = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc_aug/train_aug/label"
trainaug_file = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc_aug/train_aug/trainaug.txt"

with open(trainaug_file, 'r') as f:
    image_names = [line.strip() for line in f.readlines()]

unannotated_images = []

for image_name in image_names:
    label_path = os.path.join(label_dir, f"{image_name}.png")
    
    if not os.path.exists(label_path):
        unannotated_images.append(image_name)
        continue
    
    try:
        with Image.open(label_path) as img:
            label_array = np.array(img)
            if len(np.unique(label_array)) == 1 and np.unique(label_array)[0] == 0:
                unannotated_images.append(image_name)
    except Exception as e:
        print(f"Error processing {image_name}: {str(e)}")
        unannotated_images.append(image_name)

print(f"Total images in trainaug.txt: {len(image_names)}")
print(f"Number of effectively unannotated images: {len(unannotated_images)}")
print("Effectively unannotated images:")
for img in unannotated_images:
    print(img)

with open("effectively_unannotated_images.txt", "w") as f:
    for img in unannotated_images:
        f.write(f"{img}\n")

print("Results saved to effectively_unannotated_images.txt")