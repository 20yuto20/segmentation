import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Times New Roman に設定し、フォントサイズや図のサイズを論文向けに調整
mpl.rcParams['font.family'] = 'DejaVu Serif'
# mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.size'] = 12
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


def get_aug_name(dir_name):
    if dir_name.startswith('RA1_') and dir_name.endswith('_Randmag'):
        return dir_name.split('_')[1]
    return dir_name
def process_directory(base_dir):
    results = {'val_mIoU': {}, 'test_mIoU': {}, 'test_mIoU_std': {}}  # Added test_mIoU_std
    for seed in ['seed201', 'seed202', 'seed203']:
        voc_dir = base_dir / seed / 'voc'
        if not voc_dir.exists():
            print(f"Warning: Directory {voc_dir} does not exist. Skipping.")
            continue
        for aug_dir in voc_dir.iterdir():
            if aug_dir.is_dir():
                aug_name = get_aug_name(aug_dir.name)
                
                if aug_name not in results['val_mIoU']:
                    results['val_mIoU'][aug_name] = []
                if aug_name not in results['test_mIoU']:
                    results['test_mIoU'][aug_name] = []
                
                train_csv = aug_dir / 'train_output.csv'
                if train_csv.exists():
                    df = pd.read_csv(train_csv)
                    if 'val_mIoU' in df.columns:
                        results['val_mIoU'][aug_name].append(df['val_mIoU'].tolist())
                    else:
                        print(f"Warning: 'val_mIoU' column not found in {train_csv}")
                
                test_csv = aug_dir / 'test_output.csv'
                if test_csv.exists():
                    df = pd.read_csv(test_csv)
                    if 'test_mIoU' in df.columns and not df['test_mIoU'].empty:
                        results['test_mIoU'][aug_name].append(df['test_mIoU'].iloc[0])
                    else:
                        print(f"Warning: 'test_mIoU' column not found or empty in {test_csv}")
    
    # Average the results
    for aug_name in list(results['val_mIoU'].keys()):
        if results['val_mIoU'][aug_name]:
            results['val_mIoU'][aug_name] = np.mean(results['val_mIoU'][aug_name], axis=0).tolist()
        else:
            print(f"Warning: No valid val_mIoU data for {aug_name}. Removing from results.")
            del results['val_mIoU'][aug_name]
    
    for aug_name in list(results['test_mIoU'].keys()):
        if results['test_mIoU'][aug_name]:
            # Store the original list of values
            values = results['test_mIoU'][aug_name]
            print(f"Original test_mIoU values for {aug_name}: {values}")
            # Calculate mean and std from the original list
            results['test_mIoU'][aug_name] = np.mean(values)
            results['test_mIoU_std'][aug_name] = np.std(values)
        else:
            print(f"Warning: No valid test_mIoU data for {aug_name}. Removing from results.")
            del results['test_mIoU'][aug_name]
            del results['test_mIoU_std'][aug_name]    
    return results

def visualize_val_mIoU(results, output_dir):
    if not results['val_mIoU']:
        print("No valid val_mIoU data to visualize.")
        return

    plt.figure(figsize=(12, 6))
    for aug_name, val_mIoU in results['val_mIoU'].items():
        plt.plot(val_mIoU, label=aug_name)
    plt.title('Average Validation mIoU over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('val_mIoU')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'val_mIoU_comparison.png')
    plt.close()

def visualize_test_mIoU(results, output_dir):
    if not results['test_mIoU']:
        print("No valid test_mIoU data to visualize.")
        return

    df = pd.DataFrame({
        'test_mIoU': results['test_mIoU'],
        'test_mIoU_std': results['test_mIoU_std']
    }).reset_index()
    df.columns = ['Augmentation', 'test_mIoU', 'test_mIoU_std']
    df = df.sort_values('test_mIoU', ascending=True)  # Changed to True for bottom-to-top ordering
    print(df)
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(df['Augmentation'], df['test_mIoU'],  # Changed to barh
                   xerr=df['test_mIoU_std'],  # Changed to xerr
                   capsize=5)
    
    # plt.title('Average Test mIoU Performance')
    plt.ylabel('Augmentation')  # Swapped xlabel and ylabel
    plt.xlabel('Test mIoU')
    plt.yticks(rotation=0)  # No rotation needed for y-axis labels
    plt.xlim(0.65, 1.0)  # Changed to xlim
    
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + df['test_mIoU_std'].iloc[i] + 0.01,  # Changed x position
            bar.get_y() + bar.get_height()/2,  # Changed y position
            f'{df["test_mIoU"].iloc[i]:.4f}±{df["test_mIoU_std"].iloc[i]:.4f}',
            ha='left',  # Changed to left alignment
            va='center',  # Changed to center alignment
            rotation=0  # No rotation needed
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_mIoU_performance.pdf', bbox_inches='tight')
    plt.close()
    
    # Save rankings to CSV
    df.to_csv(output_dir / 'test_mIoU_rankings.csv', index=False)


def main():
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    base_dir = current_dir / 'output'
    output_dir = current_dir / "result" / "semseg"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = process_directory(base_dir)
    
    visualize_val_mIoU(results, output_dir)
    visualize_test_mIoU(results, output_dir)

if __name__ == '__main__':
    main()