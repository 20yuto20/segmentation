from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

def apply_autocontrast_and_debug(img_path, output_dir):
    # 画像をロード
    img = Image.open(img_path)

    # AutoContrastを適用する前のピクセル値を取得
    original_pixels = np.array(img)

    # AutoContrastを適用
    autocontrasted_img = F.autocontrast(img)

    # AutoContrastを適用した後のピクセル値を取得
    autocontrasted_pixels = np.array(autocontrasted_img)

    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 画像を保存
    original_img_path = os.path.join(output_dir, 'original_image.png')
    autocontrasted_img_path = os.path.join(output_dir, 'autocontrasted_image.png')
    diff_img_path = os.path.join(output_dir, 'difference_image.png')

    img.save(original_img_path)
    autocontrasted_img.save(autocontrasted_img_path)

    # 反転前後の画像の差分を表示
    diff_img = Image.fromarray(np.abs(original_pixels - autocontrasted_pixels))
    diff_img.save(diff_img_path)

    # 画像を表示して確認
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('AutoContrasted Image')
    plt.imshow(autocontrasted_img)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Difference Image')
    plt.imshow(diff_img)
    plt.axis('off')

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
    plt.title('Histogram of AutoContrasted Image')
    plt.hist(autocontrasted_pixels.flatten(), bins=256, color='red', alpha=0.5, label='AutoContrasted')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    autocontrasted_hist_path = os.path.join(output_dir, 'autocontrasted_histogram.png')
    plt.savefig(autocontrasted_hist_path)

    plt.show()

    # ピクセルの差分を確認
    pixel_diff = np.abs(original_pixels - autocontrasted_pixels)
    print(f"Average pixel difference: {np.mean(pixel_diff)}")


apply_autocontrast_and_debug('/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc/VOCdevkit/VOC2007/JPEGImages/000032.jpg', './output_debug_AC')


# if __name__ == '__main__':
#     apply_autocontrast_and_debug('/homes/ykohata/code/devml/homes/ypark/code/seg/dataset/voc/VOCdevkit/VOC2007/JPEGImages/000032.jpg')
