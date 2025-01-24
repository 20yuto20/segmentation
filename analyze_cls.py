import os
import pandas as pd
import numpy as np
from pathlib import Path

# ★ 修正点: matplotlib の詳細設定を行う
import matplotlib as mpl
import matplotlib.pyplot as plt

# Times New Roman に設定し、フォントサイズや図のサイズを論文向けに調整
mpl.rcParams['font.family'] = 'DejaVu Serif'
# mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.size'] = 15
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
# デフォルト図サイズ (横×縦) を論文向けに少し小さめに
mpl.rcParams['figure.figsize'] = (9.6, 6)


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
                        'test_mAP': (df['test_mAP'].dropna().tolist()[-1]
                                     if not df['test_mAP'].dropna().empty else None)
                    }
    return results

def average_results(results):
    """
    各 Augmentation (aug_name) について、複数 seed の平均と標準偏差を計算する。
    - val_mAP は epoch ごとのリスト -> epoch ごとに平均・標準偏差
    - test_mAP は seed ごとに1値 -> その平均・標準偏差
    """
    avg_results = {}
    seeds = list(results.keys())
    if not seeds:
        return avg_results

    representative_seed = seeds[0]
    for aug_name in results[representative_seed]:
        val_mAPs = [
            results[seed][aug_name]['val_mAP']
            for seed in results
            if aug_name in results[seed]
        ]
        test_mAPs = [
            results[seed][aug_name]['test_mAP']
            for seed in results
            if aug_name in results[seed] and results[seed][aug_name]['test_mAP'] is not None
        ]

        if len(val_mAPs) == 0:
            continue
        
        min_length = min(len(mAP) for mAP in val_mAPs)
        val_mAPs = [mAP[:min_length] for mAP in val_mAPs]

        val_mAP_mean = np.mean(val_mAPs, axis=0).tolist()
        val_mAP_std  = np.std(val_mAPs, axis=0).tolist()

        test_mAP_mean = np.mean(test_mAPs) if test_mAPs else None
        test_mAP_std  = np.std(test_mAPs) if test_mAPs else None

        avg_results[aug_name] = {
            'val_mAP': val_mAP_mean,
            'val_mAP_std': val_mAP_std,
            'test_mAP': test_mAP_mean,
            'test_mAP_std': test_mAP_std
        }
    
    return avg_results

def visualize_val_mAP(results, output_dir):
    # ★ 修正点: figsize は rcParams で指定してあるので省略 or 必要に応じて再設定
    plt.figure()
    for aug_name, data in results.items():
        plt.plot(data['val_mAP'], label=aug_name)
    plt.title('Average Validation mAP over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('val_mAP')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # ★ 修正点: bbox_inches='tight' で余白を最小化
    plt.savefig(output_dir / 'val_mAP_comparison.png', bbox_inches='tight')
    plt.close()

def visualize_test_mAP(results, output_dir):
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
    df = df.sort_values('test_mAP', ascending=True).reset_index(drop=True)  # Changed to True for bottom-to-top ordering

    plt.figure()
    bars = plt.barh(df['Augmentation'], df['test_mAP'],  # Changed to barh
                    xerr=df['test_mAP_std'],  # Changed to xerr
                    capsize=5)
    plt.xlim(0.75, 1.0)  # Changed to xlim
    plt.ylabel('Augmentation')  # Swapped xlabel and ylabel
    plt.xlabel('Test mAP')
    
    # No need for rotation in y-axis labels
    plt.yticks(rotation=0)

    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + df["test_mAP_std"].iloc[i] + 0.01,  # Changed x position
            bar.get_y() + bar.get_height()/2,  # Changed y position
            f'{df["test_mAP"].iloc[i]:.4f}±{df["test_mAP_std"].iloc[i]:.4f}',
            ha='left',  # Changed to left alignment
            va='center',  # Changed to center alignment
            rotation=0  # Remove rotation
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_mAP_comparison.pdf', bbox_inches='tight')
    plt.close()

    df.to_csv(output_dir / 'test_mAP_rankings.csv', index=False)    
def main():
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    base_dir = current_dir / 'output_cls'
    result_dir = current_dir / 'result' / 'cls'
    seed_dirs = ['seed201', 'seed202', 'seed203']
    
    result_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = process_directory(base_dir, seed_dirs)
    avg_results = average_results(all_results)
    
    visualize_val_mAP(avg_results, result_dir)
    visualize_test_mAP(avg_results, result_dir)

if __name__ == '__main__':
    main()
