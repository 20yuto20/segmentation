# タスク間における効果的なデータ拡張手法の研究

## プロジェクト概要
このプロジェクトはタスク間における効果的なデータ拡張手法の研究に関するものです。
今回のタスクはSemantic SegmentationとMultilabel Classificationです。
また、同一のデータセットを使用しており今回はPASACAL VOCを採用しています。
主にPyTorchを使用し、様々なデータ拡張技術、モデルアーキテクチャ、および学習戦略を実装しています。
Pytorch Lishtningを使用して実験条件を管理するようにしています。

## レポジトリのプロジェクト構造
- `bash/`: ABCIでスクリプトを実行するためのファイルが格納されています。
  - `cls/`: Multilabel Classificationのスクリプトが格納されています。`f`はABCIのfull用で`s`はG_Small用です。
  - `semseg/`: Semantic Segmentationのスクリプトが格納されています。`f`はABCIのfull用で`s`はG_Small用です。
  - `shortcut/`: fはfullのGPUを借りて、デバッグ作業するためのスクリプトです。sはG_SmallのGPUを借りて、デバッグ作業するためのスクリプトです。
- `dataset/`: データセットを格納しているディレクトリです。`.gitignore`にあるのでcottonもしくはABCI上でしか表示されません。ディレクトリの詳細は後述します。
- `local/`: ABCIでの実験結果をcottonに一時的に保存するためのディレクトリです。.gitignore`にあるのでcotton上でしか表示されません。
- `output/`: Semantic Segmentationの実験結果を保存するものとなっています。子ディレクトリはseedの値で分けられています。
- `output_cls/`: Multilabel Classificationの実験結果を保存するものとなっています。子ディレクトリはseedの値で分けられています。
- `result/`: 実験結果を可視化したものを格納しています。
- `result_v2/`: 実験結果を可視化したものを格納しています。（最新版）
- `src/`: Semantic Segmentationに関するソースコードを格納しています。詳細については`src`ディレクトリ内のREADMEを参照してください。
- `src_cls/`: Multilabel Classificationに関するソースコードを格納しています。詳細については`src_cls`ディレクトリ内のREADMEを参照してください。。
- `analyze_cls.py`: このファイルは`output_cls`内の特定のseedの結果を拾って平均を取って各データ拡張の結果を可視化するようなものになっています。可視化した結果は`result`ディレクトリ内に保存されます。
- `analyze_semeseg.py`: このファイルは`output`内の特定のseedの結果を拾って平均を取って各データ拡張の結果を可視化するようなものになっています。可視化した結果は`result`ディレクトリ内に保存されます。

## データセットについて
### 概要
今回使用したデータセットはPASCAL VOC 2012と2007です。それぞれPASCAL VOC 2012はセグメンテーションでのtrain、valに使用しました。2007の方はtestに使用しました。また、Multilabel Classificationにおいては2012のみを使用しました。

### ディレクトリ構造
```
.
├── CityScapes
│   ├── test
│   │   ├── label
│   │   ├── rgb
│   │   └── rgb_gt
│   ├── train
│   │   ├── label
│   │   ├── rgb
│   │   └── rgb_gt
│   └── val
│       ├── label
│       ├── rgb
│       └── rgb_gt
├── selected_method_affinity.csv
├── selected_method_random.csv
├── selected_method_single.csv
├── voc
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   │   ├── Annotations
│   │   │   ├── ImageSets
│   │   │   │   ├── Layout
│   │   │   │   │   ├── test.txt
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── trainval.txt
│   │   │   │   │   └── val.txt
│   │   │   │   ├── Main
│   │   │   │   │   ├── aeroplane_test.txt
│   │   │   │   │   ├── aeroplane_train.txt
│   │   │   │   │   ├── aeroplane_trainval.txt
│   │   │   │   │   ├── aeroplane_val.txt
│   │   │   │   │   ├── bicycle_test.txt
│   │   │   │   │   ├── bicycle_train.txt
│   │   │   │   │   ├── bicycle_trainval.txt
│   │   │   │   │   ├── bicycle_val.txt
│   │   │   │   │   ├── bird_test.txt
│   │   │   │   │   ├── bird_train.txt
│   │   │   │   │   ├── bird_trainval.txt
│   │   │   │   │   ├── bird_val.txt
│   │   │   │   │   ├── boat_test.txt
│   │   │   │   │   ├── boat_train.txt
│   │   │   │   │   ├── boat_trainval.txt
│   │   │   │   │   ├── boat_val.txt
│   │   │   │   │   ├── bottle_test.txt
│   │   │   │   │   ├── bottle_train.txt
│   │   │   │   │   ├── bottle_trainval.txt
│   │   │   │   │   ├── bottle_val.txt
│   │   │   │   │   ├── bus_test.txt
│   │   │   │   │   ├── bus_train.txt
│   │   │   │   │   ├── bus_trainval.txt
│   │   │   │   │   ├── bus_val.txt
│   │   │   │   │   ├── car_test.txt
│   │   │   │   │   ├── car_train.txt
│   │   │   │   │   ├── car_trainval.txt
│   │   │   │   │   ├── car_val.txt
│   │   │   │   │   ├── cat_test.txt
│   │   │   │   │   ├── cat_train.txt
│   │   │   │   │   ├── cat_trainval.txt
│   │   │   │   │   ├── cat_val.txt
│   │   │   │   │   ├── chair_test.txt
│   │   │   │   │   ├── chair_train.txt
│   │   │   │   │   ├── chair_trainval.txt
│   │   │   │   │   ├── chair_val.txt
│   │   │   │   │   ├── cow_test.txt
│   │   │   │   │   ├── cow_train.txt
│   │   │   │   │   ├── cow_trainval.txt
│   │   │   │   │   ├── cow_val.txt
│   │   │   │   │   ├── diningtable_test.txt
│   │   │   │   │   ├── diningtable_train.txt
│   │   │   │   │   ├── diningtable_trainval.txt
│   │   │   │   │   ├── diningtable_val.txt
│   │   │   │   │   ├── dog_test.txt
│   │   │   │   │   ├── dog_train.txt
│   │   │   │   │   ├── dog_trainval.txt
│   │   │   │   │   ├── dog_val.txt
│   │   │   │   │   ├── horse_test.txt
│   │   │   │   │   ├── horse_train.txt
│   │   │   │   │   ├── horse_trainval.txt
│   │   │   │   │   ├── horse_val.txt
│   │   │   │   │   ├── motorbike_test.txt
│   │   │   │   │   ├── motorbike_train.txt
│   │   │   │   │   ├── motorbike_trainval.txt
│   │   │   │   │   ├── motorbike_val.txt
│   │   │   │   │   ├── person_test.txt
│   │   │   │   │   ├── person_train.txt
│   │   │   │   │   ├── person_trainval.txt
│   │   │   │   │   ├── person_val.txt
│   │   │   │   │   ├── pottedplant_test.txt
│   │   │   │   │   ├── pottedplant_train.txt
│   │   │   │   │   ├── pottedplant_trainval.txt
│   │   │   │   │   ├── pottedplant_val.txt
│   │   │   │   │   ├── sheep_test.txt
│   │   │   │   │   ├── sheep_train.txt
│   │   │   │   │   ├── sheep_trainval.txt
│   │   │   │   │   ├── sheep_val.txt
│   │   │   │   │   ├── sofa_test.txt
│   │   │   │   │   ├── sofa_train.txt
│   │   │   │   │   ├── sofa_trainval.txt
│   │   │   │   │   ├── sofa_val.txt
│   │   │   │   │   ├── test.txt
│   │   │   │   │   ├── train_test.txt
│   │   │   │   │   ├── train_train.txt
│   │   │   │   │   ├── train_trainval.txt
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── train_val.txt
│   │   │   │   │   ├── trainval.txt
│   │   │   │   │   ├── tvmonitor_test.txt
│   │   │   │   │   ├── tvmonitor_train.txt
│   │   │   │   │   ├── tvmonitor_trainval.txt
│   │   │   │   │   ├── tvmonitor_val.txt
│   │   │   │   │   └── val.txt
│   │   │   │   └── Segmentation
│   │   │   │       ├── test.txt
│   │   │   │       ├── train.txt
│   │   │   │       ├── trainval.txt
│   │   │   │       └── val.txt
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   └── SegmentationObject
│   │   ├── VOC2012
│   │   │   ├── Annotations
│   │   │   ├── ImageSets
│   │   │   │   ├── Action
│   │   │   │   │   ├── jumping_train.txt
│   │   │   │   │   ├── jumping_trainval.txt
│   │   │   │   │   ├── jumping_val.txt
│   │   │   │   │   ├── phoning_train.txt
│   │   │   │   │   ├── phoning_trainval.txt
│   │   │   │   │   ├── phoning_val.txt
│   │   │   │   │   ├── playinginstrument_train.txt
│   │   │   │   │   ├── playinginstrument_trainval.txt
│   │   │   │   │   ├── playinginstrument_val.txt
│   │   │   │   │   ├── reading_train.txt
│   │   │   │   │   ├── reading_trainval.txt
│   │   │   │   │   ├── reading_val.txt
│   │   │   │   │   ├── ridingbike_train.txt
│   │   │   │   │   ├── ridingbike_trainval.txt
│   │   │   │   │   ├── ridingbike_val.txt
│   │   │   │   │   ├── ridinghorse_train.txt
│   │   │   │   │   ├── ridinghorse_trainval.txt
│   │   │   │   │   ├── ridinghorse_val.txt
│   │   │   │   │   ├── running_train.txt
│   │   │   │   │   ├── running_trainval.txt
│   │   │   │   │   ├── running_val.txt
│   │   │   │   │   ├── takingphoto_train.txt
│   │   │   │   │   ├── takingphoto_trainval.txt
│   │   │   │   │   ├── takingphoto_val.txt
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── trainval.txt
│   │   │   │   │   ├── usingcomputer_train.txt
│   │   │   │   │   ├── usingcomputer_trainval.txt
│   │   │   │   │   ├── usingcomputer_val.txt
│   │   │   │   │   ├── val.txt
│   │   │   │   │   ├── walking_train.txt
│   │   │   │   │   ├── walking_trainval.txt
│   │   │   │   │   └── walking_val.txt
│   │   │   │   ├── Layout
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── trainval.txt
│   │   │   │   │   └── val.txt
│   │   │   │   ├── Main
│   │   │   │   │   ├── aeroplane_train.txt
│   │   │   │   │   ├── aeroplane_trainval.txt
│   │   │   │   │   ├── aeroplane_val.txt
│   │   │   │   │   ├── bicycle_train.txt
│   │   │   │   │   ├── bicycle_trainval.txt
│   │   │   │   │   ├── bicycle_val.txt
│   │   │   │   │   ├── bird_train.txt
│   │   │   │   │   ├── bird_trainval.txt
│   │   │   │   │   ├── bird_val.txt
│   │   │   │   │   ├── boat_train.txt
│   │   │   │   │   ├── boat_trainval.txt
│   │   │   │   │   ├── boat_val.txt
│   │   │   │   │   ├── bottle_train.txt
│   │   │   │   │   ├── bottle_trainval.txt
│   │   │   │   │   ├── bottle_val.txt
│   │   │   │   │   ├── bus_train.txt
│   │   │   │   │   ├── bus_trainval.txt
│   │   │   │   │   ├── bus_val.txt
│   │   │   │   │   ├── car_train.txt
│   │   │   │   │   ├── car_trainval.txt
│   │   │   │   │   ├── car_val.txt
│   │   │   │   │   ├── cat_train.txt
│   │   │   │   │   ├── cat_trainval.txt
│   │   │   │   │   ├── cat_val.txt
│   │   │   │   │   ├── chair_train.txt
│   │   │   │   │   ├── chair_trainval.txt
│   │   │   │   │   ├── chair_val.txt
│   │   │   │   │   ├── cow_train.txt
│   │   │   │   │   ├── cow_trainval.txt
│   │   │   │   │   ├── cow_val.txt
│   │   │   │   │   ├── diningtable_train.txt
│   │   │   │   │   ├── diningtable_trainval.txt
│   │   │   │   │   ├── diningtable_val.txt
│   │   │   │   │   ├── dog_train.txt
│   │   │   │   │   ├── dog_trainval.txt
│   │   │   │   │   ├── dog_val.txt
│   │   │   │   │   ├── horse_train.txt
│   │   │   │   │   ├── horse_trainval.txt
│   │   │   │   │   ├── horse_val.txt
│   │   │   │   │   ├── motorbike_train.txt
│   │   │   │   │   ├── motorbike_trainval.txt
│   │   │   │   │   ├── motorbike_val.txt
│   │   │   │   │   ├── person_train.txt
│   │   │   │   │   ├── person_trainval.txt
│   │   │   │   │   ├── person_val.txt
│   │   │   │   │   ├── pottedplant_train.txt
│   │   │   │   │   ├── pottedplant_trainval.txt
│   │   │   │   │   ├── pottedplant_val.txt
│   │   │   │   │   ├── sheep_train.txt
│   │   │   │   │   ├── sheep_trainval.txt
│   │   │   │   │   ├── sheep_val.txt
│   │   │   │   │   ├── sofa_train.txt
│   │   │   │   │   ├── sofa_trainval.txt
│   │   │   │   │   ├── sofa_val.txt
│   │   │   │   │   ├── train_train.txt
│   │   │   │   │   ├── train_trainval.txt
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── train_val.txt
│   │   │   │   │   ├── trainval.txt
│   │   │   │   │   ├── tvmonitor_train.txt
│   │   │   │   │   ├── tvmonitor_trainval.txt
│   │   │   │   │   ├── tvmonitor_val.txt
│   │   │   │   │   └── val.txt
│   │   │   │   └── Segmentation
│   │   │   │       ├── train.txt
│   │   │   │       ├── trainval.txt
│   │   │   │       └── val.txt
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   └── SegmentationObject
│   │   └── VOC2012_test
│   │       ├── Annotations
│   │       ├── ImageSets
│   │       │   ├── Action
│   │       │   │   ├── jumping_test.txt
│   │       │   │   ├── phoning_test.txt
│   │       │   │   ├── playinginstrument_test.txt
│   │       │   │   ├── reading_test.txt
│   │       │   │   ├── ridingbike_test.txt
│   │       │   │   ├── ridinghorse_test.txt
│   │       │   │   ├── running_test.txt
│   │       │   │   ├── takingphoto_test.txt
│   │       │   │   ├── test.txt
│   │       │   │   ├── usingcomputer_test.txt
│   │       │   │   └── walking_test.txt
│   │       │   ├── Layout
│   │       │   │   └── test.txt
│   │       │   ├── Main
│   │       │   │   ├── aeroplane_test.txt
│   │       │   │   ├── bicycle_test.txt
│   │       │   │   ├── bird_test.txt
│   │       │   │   ├── boat_test.txt
│   │       │   │   ├── bottle_test.txt
│   │       │   │   ├── bus_test.txt
│   │       │   │   ├── car_test.txt
│   │       │   │   ├── cat_test.txt
│   │       │   │   ├── chair_test.txt
│   │       │   │   ├── cow_test.txt
│   │       │   │   ├── diningtable_test.txt
│   │       │   │   ├── dog_test.txt
│   │       │   │   ├── horse_test.txt
│   │       │   │   ├── motorbike_test.txt
│   │       │   │   ├── person_test.txt
│   │       │   │   ├── pottedplant_test.txt
│   │       │   │   ├── sheep_test.txt
│   │       │   │   ├── sofa_test.txt
│   │       │   │   ├── test.txt
│   │       │   │   ├── train_test.txt
│   │       │   │   └── tvmonitor_test.txt
│   │       │   └── Segmentation
│   │       │       └── test.txt
│   │       └── JPEGImages
│   └── VOCtrainval_11-May-2012.tar
├── voc_aug
│   ├── test
│   │   ├── image
│   │   └── test.txt
│   ├── test_2007
│   │   ├── image
│   │   ├── label
│   │   └── test.txt
│   ├── train_aug
│   │   ├── image
│   │   ├── label
│   │   └── trainaug.txt
│   └── val
│       ├── image
│       ├── label
│       └── val.txt
├── voc_datasets.tar.gz
└── vocsbd
    ├── SBD
    │   ├── benchmark_RELEASE
    │   │   ├── benchmark_code_RELEASE
    │   │   │   ├── benchmark_category.m
    │   │   │   ├── category_names.m
    │   │   │   ├── collect_eval_bdry.m
    │   │   │   ├── config.m
    │   │   │   ├── cp_src
    │   │   │   │   ├── Array.hh
    │   │   │   │   ├── build.m
    │   │   │   │   ├── build.sh
    │   │   │   │   ├── correspondPixels.cc
    │   │   │   │   ├── correspondPixels.mexa64
    │   │   │   │   ├── csa.cc
    │   │   │   │   ├── csa_defs.h
    │   │   │   │   ├── csa.hh
    │   │   │   │   ├── csa_types.h
    │   │   │   │   ├── Exception.cc
    │   │   │   │   ├── Exception.hh
    │   │   │   │   ├── kofn.cc
    │   │   │   │   ├── kofn.hh
    │   │   │   │   ├── match.cc
    │   │   │   │   ├── match.hh
    │   │   │   │   ├── Matrix.cc
    │   │   │   │   ├── Matrix.hh
    │   │   │   │   ├── Point.hh
    │   │   │   │   ├── Random.cc
    │   │   │   │   ├── Random.hh
    │   │   │   │   ├── README
    │   │   │   │   ├── Sort.hh
    │   │   │   │   ├── String.cc
    │   │   │   │   ├── String.hh
    │   │   │   │   ├── Timer.cc
    │   │   │   │   └── Timer.hh
    │   │   │   ├── create_isoF_figure.m
    │   │   │   ├── demo
    │   │   │   │   ├── datadir
    │   │   │   │   │   ├── cls
    │   │   │   │   │   ├── img
    │   │   │   │   │   ├── inst
    │   │   │   │   │   ├── train.txt
    │   │   │   │   │   └── val.txt
    │   │   │   │   ├── demo_config.m
    │   │   │   │   ├── indir
    │   │   │   │   │   ├── 2008_000051.bmp
    │   │   │   │   │   └── 2008_000195.bmp
    │   │   │   │   └── outdir
    │   │   │   │       ├── 2008_000051_ev1.txt
    │   │   │   │       ├── 2008_000195_ev1.txt
    │   │   │   │       ├── eval_bdry_img.txt
    │   │   │   │       ├── eval_bdry_thr.txt
    │   │   │   │       └── eval_bdry.txt
    │   │   │   ├── evaluate_bdry.m
    │   │   │   ├── evaluate_bmps.m
    │   │   │   ├── isoF.fig
    │   │   │   ├── plot_eval.m
    │   │   │   ├── plot_eval_multiple.m
    │   │   │   ├── run_demo.m
    │   │   │   └── seg2bdry.m
    │   │   ├── BharathICCV2011.pdf
    │   │   ├── dataset
    │   │   └── README
    │   ├── benchmark.tgz
    │   ├── inst
    │   ├── train_comp.txt
    │   ├── train_noval.txt
    │   ├── train.txt
    │   ├── trainval.txt
    │   └── val.txt
    ├── trainaug.txt
    ├── VOC2007
    │   ├── VOCdevkit
    │   │   └── VOC2007
    │   │       ├── Annotations
    │   │       ├── ImageSets
    │   │       │   ├── Layout
    │   │       │   │   └── test.txt
    │   │       │   ├── Main
    │   │       │   │   ├── aeroplane_test.txt
    │   │       │   │   ├── bicycle_test.txt
    │   │       │   │   ├── bird_test.txt
    │   │       │   │   ├── boat_test.txt
    │   │       │   │   ├── bottle_test.txt
    │   │       │   │   ├── bus_test.txt
    │   │       │   │   ├── car_test.txt
    │   │       │   │   ├── cat_test.txt
    │   │       │   │   ├── chair_test.txt
    │   │       │   │   ├── cow_test.txt
    │   │       │   │   ├── diningtable_test.txt
    │   │       │   │   ├── dog_test.txt
    │   │       │   │   ├── horse_test.txt
    │   │       │   │   ├── motorbike_test.txt
    │   │       │   │   ├── person_test.txt
    │   │       │   │   ├── pottedplant_test.txt
    │   │       │   │   ├── sheep_test.txt
    │   │       │   │   ├── sofa_test.txt
    │   │       │   │   ├── test.txt
    │   │       │   │   ├── train_test.txt
    │   │       │   │   └── tvmonitor_test.txt
    │   │       │   └── Segmentation
    │   │       │       └── test.txt
    │   │       ├── JPEGImages
    │   │       ├── SegmentationClass
    │   │       └── SegmentationObject
    │   └── VOCtest_06-Nov-2007.tar
    ├── VOC2012
    │   ├── JPEGImages
    │   ├── SegmentationClass
    │   ├── SegmentationClassAug
    │   │   └── aug.txt
    │   ├── train.txt
    │   ├── val.txt
    │   ├── VOCdevkit
    │   │   └── VOC2012
    │   │       ├── Annotations
    │   │       ├── ImageSets
    │   │       │   ├── Action
    │   │       │   │   ├── jumping_train.txt
    │   │       │   │   ├── jumping_trainval.txt
    │   │       │   │   ├── jumping_val.txt
    │   │       │   │   ├── phoning_train.txt
    │   │       │   │   ├── phoning_trainval.txt
    │   │       │   │   ├── phoning_val.txt
    │   │       │   │   ├── playinginstrument_train.txt
    │   │       │   │   ├── playinginstrument_trainval.txt
    │   │       │   │   ├── playinginstrument_val.txt
    │   │       │   │   ├── reading_train.txt
    │   │       │   │   ├── reading_trainval.txt
    │   │       │   │   ├── reading_val.txt
    │   │       │   │   ├── ridingbike_train.txt
    │   │       │   │   ├── ridingbike_trainval.txt
    │   │       │   │   ├── ridingbike_val.txt
    │   │       │   │   ├── ridinghorse_train.txt
    │   │       │   │   ├── ridinghorse_trainval.txt
    │   │       │   │   ├── ridinghorse_val.txt
    │   │       │   │   ├── running_train.txt
    │   │       │   │   ├── running_trainval.txt
    │   │       │   │   ├── running_val.txt
    │   │       │   │   ├── takingphoto_train.txt
    │   │       │   │   ├── takingphoto_trainval.txt
    │   │       │   │   ├── takingphoto_val.txt
    │   │       │   │   ├── train.txt
    │   │       │   │   ├── trainval.txt
    │   │       │   │   ├── usingcomputer_train.txt
    │   │       │   │   ├── usingcomputer_trainval.txt
    │   │       │   │   ├── usingcomputer_val.txt
    │   │       │   │   ├── val.txt
    │   │       │   │   ├── walking_train.txt
    │   │       │   │   ├── walking_trainval.txt
    │   │       │   │   └── walking_val.txt
    │   │       │   ├── Layout
    │   │       │   │   ├── train.txt
    │   │       │   │   ├── trainval.txt
    │   │       │   │   └── val.txt
    │   │       │   ├── Main
    │   │       │   │   ├── aeroplane_train.txt
    │   │       │   │   ├── aeroplane_trainval.txt
    │   │       │   │   ├── aeroplane_val.txt
    │   │       │   │   ├── bicycle_train.txt
    │   │       │   │   ├── bicycle_trainval.txt
    │   │       │   │   ├── bicycle_val.txt
    │   │       │   │   ├── bird_train.txt
    │   │       │   │   ├── bird_trainval.txt
    │   │       │   │   ├── bird_val.txt
    │   │       │   │   ├── boat_train.txt
    │   │       │   │   ├── boat_trainval.txt
    │   │       │   │   ├── boat_val.txt
    │   │       │   │   ├── bottle_train.txt
    │   │       │   │   ├── bottle_trainval.txt
    │   │       │   │   ├── bottle_val.txt
    │   │       │   │   ├── bus_train.txt
    │   │       │   │   ├── bus_trainval.txt
    │   │       │   │   ├── bus_val.txt
    │   │       │   │   ├── car_train.txt
    │   │       │   │   ├── car_trainval.txt
    │   │       │   │   ├── car_val.txt
    │   │       │   │   ├── cat_train.txt
    │   │       │   │   ├── cat_trainval.txt
    │   │       │   │   ├── cat_val.txt
    │   │       │   │   ├── chair_train.txt
    │   │       │   │   ├── chair_trainval.txt
    │   │       │   │   ├── chair_val.txt
    │   │       │   │   ├── cow_train.txt
    │   │       │   │   ├── cow_trainval.txt
    │   │       │   │   ├── cow_val.txt
    │   │       │   │   ├── diningtable_train.txt
    │   │       │   │   ├── diningtable_trainval.txt
    │   │       │   │   ├── diningtable_val.txt
    │   │       │   │   ├── dog_train.txt
    │   │       │   │   ├── dog_trainval.txt
    │   │       │   │   ├── dog_val.txt
    │   │       │   │   ├── horse_train.txt
    │   │       │   │   ├── horse_trainval.txt
    │   │       │   │   ├── horse_val.txt
    │   │       │   │   ├── motorbike_train.txt
    │   │       │   │   ├── motorbike_trainval.txt
    │   │       │   │   ├── motorbike_val.txt
    │   │       │   │   ├── person_train.txt
    │   │       │   │   ├── person_trainval.txt
    │   │       │   │   ├── person_val.txt
    │   │       │   │   ├── pottedplant_train.txt
    │   │       │   │   ├── pottedplant_trainval.txt
    │   │       │   │   ├── pottedplant_val.txt
    │   │       │   │   ├── sheep_train.txt
    │   │       │   │   ├── sheep_trainval.txt
    │   │       │   │   ├── sheep_val.txt
    │   │       │   │   ├── sofa_train.txt
    │   │       │   │   ├── sofa_trainval.txt
    │   │       │   │   ├── sofa_val.txt
    │   │       │   │   ├── train_train.txt
    │   │       │   │   ├── train_trainval.txt
    │   │       │   │   ├── train.txt
    │   │       │   │   ├── train_val.txt
    │   │       │   │   ├── trainval.txt
    │   │       │   │   ├── tvmonitor_train.txt
    │   │       │   │   ├── tvmonitor_trainval.txt
    │   │       │   │   ├── tvmonitor_val.txt
    │   │       │   │   └── val.txt
    │   │       │   └── Segmentation
    │   │       │       ├── train.txt
    │   │       │       ├── trainval.txt
    │   │       │       └── val.txt
    │   │       ├── JPEGImages
    │   │       ├── SegmentationClass
    │   │       └── SegmentationObject
    │   └── VOCtrainval_11-May-2012.tar
    └── VOCSBD
        ├── JPEGImages
        ├── SegmentationClass
        └── split
            ├── train.txt
            └── val.txt
```


