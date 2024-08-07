def read_file(file_path):
    with open(file_path, 'r') as file:
        return set(file.read().splitlines())

# ファイルパスを定義
sbd_trainval_path = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc/VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt"
voc_test_path = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc_aug/test/test.txt"

# ファイルの内容を読み込む
sbd_trainval = read_file(sbd_trainval_path)
voc_test = read_file(voc_test_path)

# SBD trainvalセットがVOC testセットを完全に内包しているか確認
is_subset = voc_test.issubset(sbd_trainval)

print(f"SBD trainval set size: {len(sbd_trainval)}")
print(f"VOC test set size: {len(voc_test)}")

if is_subset:
    print("VOC test set is completely included in SBD trainval set.")
else:
    print("VOC test set is NOT completely included in SBD trainval set.")
    
    # 内包されていない要素を表示
    not_included = voc_test - sbd_trainval
    print(f"Number of elements in VOC test set not included in SBD trainval: {len(not_included)}")
    print("Sample of not included elements (up to 10):")
    for item in list(not_included)[:10]:
        print(item)

# 重複している要素の数を表示
intersection = voc_test.intersection(sbd_trainval)
print(f"\nNumber of elements common to both sets: {len(intersection)}")
print(f"Percentage of VOC test set included in SBD trainval: {len(intersection) / len(voc_test) * 100:.2f}%")