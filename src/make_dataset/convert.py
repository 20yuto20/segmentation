import os
import numpy as np
from scipy.io import loadmat
from PIL import Image

def convert_mat_to_png(input_dir, output_dir):
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 入力ディレクトリ内の全.matファイルを処理
    for filename in os.listdir(input_dir):
        if filename.endswith('.mat'):
            mat_path = os.path.join(input_dir, filename)
            png_path = os.path.join(output_dir, filename.replace('.mat', '.png'))

            # .matファイルを読み込む
            mat_contents = loadmat(mat_path)
            
            # 'GTcls' キーの下の 'Segmentation' フィールドからデータを取得
            segmentation_data = mat_contents['GTcls']['Segmentation'][0][0]

            # NumPy配列をPIL Imageに変換
            img = Image.fromarray(segmentation_data.astype(np.uint8))

            # PNG形式で保存
            img.save(png_path)

            print(f"Converted {filename} to PNG")

    print("Conversion completed.")

# 入力と出力のディレクトリパスを設定
input_directory = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc_aug/train_aug/cls"
output_directory = "/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc_aug/train_aug/label"

# 変換を実行
convert_mat_to_png(input_directory, output_directory)