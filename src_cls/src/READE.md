# 画像分類のための機械学習プロジェクト / Machine Learning Project for Image Classification

## 日本語

### プロジェクト概要
このプロジェクトは、最新の機械学習技術を用いた画像分類タスクのための包括的なフレームワークを提供します。主にPyTorchを使用し、様々なデータ拡張技術、モデルアーキテクチャ、および学習戦略を実装しています。

### 主な機能
1. **データローディングとデータ拡張**: カスタマイズ可能なデータローダーと高度なデータ拡張技術（RandAugmentを含む）を実装。
2. **モデルアーキテクチャ**: ResNet、WideResNet、VGGなどの様々なモデルアーキテクチャをサポート。
3. **学習と評価**: トレーニング、検証、テストのための包括的なパイプラインを提供。
4. **設定管理**: OmegaConfを使用した柔軟で拡張性の高い設定管理システム。
5. **可視化**: 学習曲線やデータ拡張の効果を可視化するためのツール。

### セットアップ
1. リポジトリをクローン:
   ```
   git clone [repository_url]
   ```
2. 依存関係をインストール:
   ```
   pip install -r requirements.txt
   ```

### 使用方法
1. 設定ファイルを編集 (`src/conf/`)
2. メインスクリプトを実行:
   ```
   python src/main.py [config_name]
   ```

### プロジェクト構造
- `src/`: ソースコード
  - `conf/`: 設定ファイル
  - `models/`: モデル定義
  - `utils/`: ユーティリティ関数
  - `main.py`: メイン実行スクリプト
- `scripts/`: 実行スクリプト
- `setup/`: セットアップスクリプト

### 主要コンポーネント
- `dataloader.py`: データセットのロードと前処理
- `train_val.py`: トレーニングと評価ループ
- `augment.py` & `randaugment.py`: データ拡張技術
- `set_cfg.py`: 設定管理
- `affinity.py`: アフィニティ計算（データ拡張の効果測定）

### 貢献
プルリクエストは歓迎します。大きな変更を加える場合は、まずissueを開いて議論してください。

### ライセンス
[ライセンス情報を記載]

## English

### Project Overview
This project provides a comprehensive framework for image classification tasks using state-of-the-art machine learning techniques. It primarily uses PyTorch and implements various data augmentation techniques, model architectures, and training strategies.

### Key Features
1. **Data Loading and Augmentation**: Implements customizable data loaders and advanced data augmentation techniques, including RandAugment.
2. **Model Architectures**: Supports various model architectures including ResNet, WideResNet, and VGG.
3. **Training and Evaluation**: Provides a comprehensive pipeline for training, validation, and testing.
4. **Configuration Management**: Flexible and extensible configuration management system using OmegaConf.
5. **Visualization**: Tools for visualizing learning curves and the effects of data augmentation.

### Setup
1. Clone the repository:
   ```
   git clone [repository_url]
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage
1. Edit the configuration file (`src/conf/`)
2. Run the main script:
   ```
   python src/main.py [config_name]
   ```

### Project Structure
- `src/`: Source code
  - `conf/`: Configuration files
  - `models/`: Model definitions
  - `utils/`: Utility functions
  - `main.py`: Main execution script
- `scripts/`: Execution scripts
- `setup/`: Setup scripts

### Key Components
- `dataloader.py`: Dataset loading and preprocessing
- `train_val.py`: Training and evaluation loops
- `augment.py` & `randaugment.py`: Data augmentation techniques
- `set_cfg.py`: Configuration management
- `affinity.py`: Affinity calculation (measuring the effect of data augmentation)

### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

### License
[Insert License Information]