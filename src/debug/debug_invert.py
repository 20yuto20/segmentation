from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import os

def apply_invert_and_debug(img_path, output_dir):
    # 画像をロード
    img = Image.open(img_path)

    # Invertを適用する前のピクセル値を取得
    original_pixels = np.array(img)

    # Invertを適用
    inverted_img = ImageOps.invert(img)

    # Invertを適用した後のピクセル値を取得
    inverted_pixels = np.array(inverted_img)

    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 画像を保存
    original_img_path = os.path.join(output_dir, 'original_image.png')
    inverted_img_path = os.path.join(output_dir, 'inverted_image.png')
    diff_img_path = os.path.join(output_dir, 'difference_image.png')

    img.save(original_img_path)
    inverted_img.save(inverted_img_path)

    # 反転前後の画像の差分を表示
    diff_img = Image.fromarray(np.abs(original_pixels - inverted_pixels))
    diff_img.save(diff_img_path)

    # 画像を表示して確認
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Inverted Image')
    plt.imshow(inverted_img)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Difference Image')
    plt.imshow(diff_img)
    plt.axis('off')

    plt.show()

    # ピクセルの差分を確認
    pixel_diff = np.abs(original_pixels - inverted_pixels)
    print(f"Average pixel difference: {np.mean(pixel_diff)}")

    # ヒストグラムを保存
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.title('Histogram of Original Image')
    plt.hist(original_pixels.flatten(), bins=256, color='blue', alpha=0.5, label='Original')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    original_hist_path = os.path.join(output_dir, 'original_histogram.png')
    plt.savefig(original_hist_path)

    plt.subplot(1, 2, 2)
    plt.title('Histogram of Inverted Image')
    plt.hist(inverted_pixels.flatten(), bins=256, color='red', alpha=0.5, label='Inverted')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    inverted_hist_path = os.path.join(output_dir, 'inverted_histogram.png')
    plt.savefig(inverted_hist_path)

    plt.show()

apply_invert_and_debug('/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc/VOCdevkit/VOC2007/JPEGImages/000032.jpg', './output_debug_invert')
