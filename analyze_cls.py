import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def process_directory(base_dir, seed_dirs):
    results = {seed: {} for seed in seed_dirs}
    for seed in seed_dirs:
        voc_dir = base_dir / seed / 'voc'
        for aug_dir in voc_dir.iterdir():
            if aug_dir.is_dir():
                # RA1_XXXXX_Randmag の場合は一部文字列処理
                if aug_dir.name.startswith('RA1_') and aug_dir.name.endswith('_Randmag'):
                    aug_name = aug_dir.name.split('_')[1]
                else:
                    aug_name = aug_dir.name

                csv_path = aug_dir / 'output.csv'
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    results[seed][aug_name] = {
                        'val_mAP': df['val_mAP'].tolist(),
                        # 推論速度を削除したので 'avg_inference_time' は削除
                        'test_mAP': df['test_mAP'].dropna().tolist()[-1] 
                                    if not df['test_mAP'].dropna().empty else None
                    }
    return results

def average_results(results):
    """
    各 Augmentation (aug_name) について、複数 seed の平均と標準偏差を計算する。
    - val_mAP は epoch ごとのリスト -> epoch ごとに平均・標準偏差を求める
    - test_mAP は seed ごとに1値 -> その平均・標準偏差を求める
    """
    avg_results = {}
    
    seeds = list(results.keys())
    if not seeds:
        return avg_results

    # 代表的に最初の seed に含まれる augmentation の名前を基準にする
    representative_seed = seeds[0]
    for aug_name in results[representative_seed]:
        # val_mAP は各 seed で epoch ごとの list
        val_mAPs = [
            results[seed][aug_name]['val_mAP']
            for seed in results
            if aug_name in results[seed]
        ]
        # test_mAP は各 seed で 1 値
        test_mAPs = [
            results[seed][aug_name]['test_mAP']
            for seed in results
            if aug_name in results[seed] and results[seed][aug_name]['test_mAP'] is not None
        ]

        if len(val_mAPs) == 0:
            continue
        
        # epoch 数を合わせる
        min_length = min(len(mAP) for mAP in val_mAPs)
        val_mAPs = [mAP[:min_length] for mAP in val_mAPs]

        # val_mAP の平均・標準偏差 (epoch ごと)
        val_mAP_mean = np.mean(val_mAPs, axis=0).tolist()
        val_mAP_std  = np.std(val_mAPs, axis=0).tolist()  # <-- 標準偏差を追加

        # test_mAP の平均・標準偏差 (seed ごと)
        test_mAP_mean = np.mean(test_mAPs) if test_mAPs else None
        test_mAP_std  = np.std(test_mAPs) if test_mAPs else None  # <-- 標準偏差を追加

        avg_results[aug_name] = {
            'val_mAP': val_mAP_mean,
            'val_mAP_std': val_mAP_std,       # <-- 追加
            'test_mAP': test_mAP_mean,
            'test_mAP_std': test_mAP_std      # <-- 追加
        }
    
    return avg_results

def visualize_val_mAP(results, output_dir):
    plt.figure(figsize=(12, 6))
    for aug_name, data in results.items():
        plt.plot(data['val_mAP'], label=aug_name)
    plt.title('Average Validation mAP over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('val_mAP')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'val_mAP_comparison.png')
    plt.close()

def visualize_test_mAP(results, output_dir):
    """
    test_mAP (平均) の棒グラフを表示し、
    y 軸を 0.75 ~ 1.0 に固定する。
    標準偏差 test_mAP_std も DataFrame に格納し、CSV に出力する。
    """
    aug_names = []
    test_mAP_means = []
    test_mAP_stds = []

    for aug_name, data in results.items():
        if data['test_mAP'] is not None:
            aug_names.append(aug_name)
            test_mAP_means.append(data['test_mAP'])
            test_mAP_stds.append(data['test_mAP_std'])

    if not test_mAP_means:
        return

    df = pd.DataFrame({
        'Augmentation': aug_names,
        'test_mAP': test_mAP_means,
        'test_mAP_std': test_mAP_stds
    })
    # test_mAP の大きい順にソート
    df = df.sort_values('test_mAP', ascending=False).reset_index(drop=True)

    # 棒グラフを生成 (y 軸 0.75 ~ 1.0)
    plt.figure(figsize=(14, 8))
    bars = plt.bar(df['Augmentation'], df['test_mAP'])
    plt.ylim(0.75, 1.0)  # <-- y 軸の下限を0.75, 上限を1.0に固定
    plt.title('Average Test mAP')
    plt.xlabel('Augmentation')
    plt.ylabel('Test mAP')
    plt.xticks(rotation=90)

    # バーの上に値を表示
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f'{df["test_mAP"].iloc[i]:.4f}',
            ha='center',
            va='bottom',
            rotation=90
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_mAP_comparison.png')
    plt.close()

    # CSV 出力
    df.to_csv(output_dir / 'test_mAP_rankings.csv', index=False)

def main():
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    base_dir = current_dir / 'output_cls'
    result_dir = current_dir / 'result' / 'cls'
    seed_dirs = ['seed201', 'seed202', 'seed203']
    
    result_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = process_directory(base_dir, seed_dirs)
    avg_results = average_results(all_results)
    
    # 可視化
    visualize_val_mAP(avg_results, result_dir)
    visualize_test_mAP(avg_results, result_dir)
    # 推論速度関連機能は削除したため呼び出しもしない

if __name__ == '__main__':
    main()
