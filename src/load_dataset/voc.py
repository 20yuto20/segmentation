import os
import urllib.request
import tarfile
import numpy as np

import os.path as osp
from PIL import Image
import torch.utils.data as data
import torch

from .psp_data_augmentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor

############################
## datsetはHPより以下のコードを使って事前にダウンロード ######

# フォルダ「data」が存在しない場合は作成する
#######　適宜修正してください  ##################
# data_dir = "/homes/ypark/code/dataset"
# ## abci用
# data_dir = "/groups/gaa50073/park-yuna/datasets"
# if not os.path.exists(data_dir):
#     os.mkdir(data_dir)


# # 公式のHOからVOC2012のデータセットをダウンロード
# # 時間がかかります（約15分）
# url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
# url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"

# target_path = os.path.join(data_dir, "VOCtrainval_11-May-2012.tar") 
# target_path = os.path.join(data_dir, "VOCtest_06-Nov-2007.tar") 

# if not os.path.exists(target_path):
#     urllib.request.urlretrieve(url, target_path)
    
#     tar = tarfile.TarFile(target_path)  # tarファイルを読み込み
#     tar.extractall(data_dir)  # tarを解凍
#     tar.close()  # tarファイルをクローズ
    
############################

# "/groups/gaa50073/park-yuna/datasets/VOCtest_06-Nov-2007.tar"
# "/groups/gaa50073/park-yuna/datasets/VOCtrainval_11-May-2012.tar"
# を解凍すると，
# "/groups/gaa50073/park-yuna/datasets/VOCdevkit/ - VOC2007 と - VOC2012 ができる"

# cfg.default.dataset_dir :  (SGE_LOCAL_DIR) + dataset/



# TODO: testを読み出すものを追記する、rootpathがtraivalを受け取っているのでtest用のパスを作る
# どの画像がtrain, valにそれぞれ含まれるかを指定したtxtファイルから画像名のリストを取得
def make_datapath_list(path_2012, path_2007):
    """
    学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。

    Parameters
    ----------
    rootpath : str
        データフォルダへのパス

    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    """

    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_template = osp.join(path_2012, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(path_2012, 'SegmentationClass', '%s.png')

    test_imgpath_template = osp.join(path_2007, 'JPEGImages', '%s.jpg')
    test_annopath_template = osp.join(path_2007, 'SegmentationClass', '%s.png')

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    # ここにどのデータがtrainでどれがvalか書いてある
    train_id_names = osp.join(path_2012 + 'ImageSets/Segmentation/train.txt')
    val_id_names = osp.join(path_2012 + 'ImageSets/Segmentation/val.txt')
    test_id_names = osp.join(path_2007 + 'ImageSets/Segmentation/test.txt')

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)  # 画像のパス
        anno_path = (annopath_template % file_id)  # アノテーションのパス
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)  # 画像のパス
        anno_path = (annopath_template % file_id)  # アノテーションのパス
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    # テストデータの画像ファイルとアノテーションファイルへのパスリストを作成
    test_img_list = list()
    test_anno_list = list()

    for line in open(test_id_names):
        file_id = line.strip()
        img_path = (test_imgpath_template % file_id)
        anno_path = (test_annopath_template % file_id)
        test_img_list.append(img_path)
        test_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list, test_img_list, test_anno_list



# TODO: testにも対応した仕様に変更する
class VOCDataset(data.Dataset):
    """
    VOC2012のDatasetを作成するクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス
    """

    def __init__(self, img_list, anno_list, phase, transform, img_size):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとアノテーションを取得
        '''
        img, anno_class_img = self.pull_item(index)
        sample = {'image': img, 'label': anno_class_img}

        # 3. データ拡張を実施
        if self.transform:
            sample = self.transform(sample)

        return sample



    def pull_item(self, index):
        '''画像のTensor形式のデータ、アノテーションを取得する'''

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [高さ][幅][色RGB]

        # 2. アノテーション画像読み込み
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)   # [高さ][幅]

        ### データごとにサイズが違うので均一のサイズにリサイズ
        resize_fn = Resize(self.img_size)
        img, anno_class_img = resize_fn(img, anno_class_img)

        # # 3. 前処理を実施
        # img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img
    



## どんなデータセットか確かめたいときに使用 ##

# rootpath = "/homes/ypark/code/dataset/VOCdevkit/VOC2012/"
# train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
#     rootpath=rootpath)

# print(train_img_list[0])
# print(train_anno_list[0])

# # (RGB)の色の平均値と標準偏差
# color_mean = (0.485, 0.456, 0.406)
# color_std = (0.229, 0.224, 0.225)

# # データセット作成
# train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
#     input_size=475, color_mean=color_mean, color_std=color_std))

# val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
#     input_size=475, color_mean=color_mean, color_std=color_std))

# # データの取り出し例

# # jpeg画像には 縦，横，チャンネル数(RGB)のデータ torch.Size([3, 475, 475])
# # png画像 (正解ラベル) には 縦，横， torch.Size([475, 475]) ，1ピクセルにスカラー値
# # annoは0-20までのクラス，背景は0

# print(val_dataset.__getitem__(0)[0].shape)
# print(val_dataset.__getitem__(0)[1].shape)
# print(val_dataset.__getitem__(10)[1])
# for i in range(10):
#     anno_ex = val_dataset.__getitem__(i)[1]
#     max_value = torch.max(anno_ex)
#     min_value = torch.min(anno_ex)
#     print(f"最大値: {max_value}, 最小値: {min_value}")
#     print(f"unieque anno classes : {torch.unique(anno_ex)}")






##### PSPNetで使われているtransforms ** 参考までに #######
class DataTransform():
    """
    画像とアノテーションの前処理クラス。訓練時と検証時で異なる動作をする。
    画像のサイズをinput_size x input_sizeにする。
    訓練時はデータオーギュメンテーションする。


    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ。
    color_mean : (R, G, B)
        各色チャネルの平均値。
    color_std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                Scale(scale=[0.5, 1.5]),  # 画像の拡大
                RandomRotation(angle=[-10, 10]),  # 回転
                RandomMirror(),  # ランダムミラー
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ]),
            'val': Compose([
                Resize(input_size),  # リサイズ(input_size)
                Normalize_Tensor(color_mean, color_std)  # 色情報の標準化とテンソル化
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img, anno_class_img)


