import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_aug_name(dir_name):
    if dir_name.startswith('RA2_') and dir_name.endswith('_Randmag'):
        return dir_name.split('_')[1]
    return dir_name

def process_directory(base_dir):
    results = {'val_mIoU': {}, 'test_mIoU': {}}
    for seed in ['seed2024', 'seed2025', 'seed2026']:
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
            results['test_mIoU'][aug_name] = np.mean(results['test_mIoU'][aug_name])
        else:
            print(f"Warning: No valid test_mIoU data for {aug_name}. Removing from results.")
            del results['test_mIoU'][aug_name]
    
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

    df = pd.DataFrame.from_dict(results['test_mIoU'], orient='index', columns=['test_mIoU'])
    df = df.sort_values('test_mIoU', ascending=False).reset_index()
    df.columns = ['Augmentation', 'test_mIoU']
    
    df['Relative Performance'] = (df['test_mIoU'] - df['test_mIoU'].min()) / (df['test_mIoU'].max() - df['test_mIoU'].min())
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(df['Augmentation'], df['Relative Performance'])
    plt.title('Relative Average Test mIoU Performance')
    plt.xlabel('Augmentation')
    plt.ylabel('Relative Performance')
    plt.xticks(rotation=90)
    
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f'{df["test_mIoU"].iloc[i]:.4f}', 
                 ha='center', va='bottom', rotation=90)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_mIoU_relative_performance.png')
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