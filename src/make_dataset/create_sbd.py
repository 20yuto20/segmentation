import os
import shutil

def read_file(file_path):
    with open(file_path, 'r') as file:
        return set(file.read().splitlines())

def write_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write('\n'.join(content))

# Define paths
base_path = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/vocsbd"
voc_path = os.path.join(base_path, "VOC2012")
sbd_path = os.path.join(base_path, "SBD")
output_path = os.path.join(base_path, "trainaug.txt")

# Read VOC2012 train set
voc_train_path = os.path.join(voc_path, "train.txt")
voc_train = read_file(voc_train_path)

# Read SBD train and val sets
sbd_train_path = os.path.join(sbd_path, "train.txt")
sbd_val_path = os.path.join(sbd_path, "val.txt")
sbd_train = read_file(sbd_train_path)
sbd_val = read_file(sbd_val_path)

# Combine all sets
trainaug = voc_train.union(sbd_train).union(sbd_val)

print(f"VOC2012 train set size: {len(voc_train)}")
print(f"SBD train set size: {len(sbd_train)}")
print(f"SBD val set size: {len(sbd_val)}")
print(f"Combined trainaug set size: {len(trainaug)}")

if len(trainaug) != 10582:
    print(f"Warning: trainaug set size ({len(trainaug)}) is not 10,582 as expected.")
else:
    print("Successfully created trainaug set with 10,582 images.")

# Sort and write the trainaug set
sorted_trainaug = sorted(trainaug)
write_file(output_path, sorted_trainaug)
print(f"Created {output_path} with {len(sorted_trainaug)} entries.")

# Copy SegmentationClassAug to VOC2012 folder
seg_aug_src = os.path.join(sbd_path, "SegmentationClassAug")
seg_aug_dst = os.path.join(voc_path, "SegmentationClassAug")
if not os.path.exists(seg_aug_dst):
    shutil.copytree(seg_aug_src, seg_aug_dst)
    print(f"Copied SegmentationClassAug from {seg_aug_src} to {seg_aug_dst}")
else:
    print("SegmentationClassAug already exists in VOC2012 folder.")

# Output some sample entries for verification
print("\nSample entries from the trainaug set:")
for entry in sorted_trainaug[:10]:
    print(entry)