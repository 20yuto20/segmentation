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
                # Skip RA2_Random_Randmag directory
                if aug_dir.name == 'RA2_Random_Randmag':
                    continue
                
                if aug_dir.name.startswith('RA2_') and aug_dir.name.endswith('_Randmag'):
                    aug_name = aug_dir.name.split('_')[1]
                else:
                    aug_name = aug_dir.name
                
                csv_path = aug_dir / 'output.csv'
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    results[seed][aug_name] = {
                        'val_mAP': df['val_mAP'].tolist(),
                        'test_mAP': df['test_mAP'].dropna().tolist()[-1] if not df['test_mAP'].dropna().empty else None
                    }
    return results

def average_results(results):
    avg_results = {}
    for aug_name in results[list(results.keys())[0]]:
        val_mAPs = [results[seed][aug_name]['val_mAP'] for seed in results if aug_name in results[seed]]
        test_mAPs = [results[seed][aug_name]['test_mAP'] for seed in results if aug_name in results[seed] and results[seed][aug_name]['test_mAP'] is not None]
        
        # Ensure all val_mAPs have the same length
        min_length = min(len(mAP) for mAP in val_mAPs)
        val_mAPs = [mAP[:min_length] for mAP in val_mAPs]
        
        avg_results[aug_name] = {
            'val_mAP': np.mean(val_mAPs, axis=0).tolist(),
            'test_mAP': np.mean(test_mAPs) if test_mAPs else None
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
    aug_names = list(results.keys())
    test_mAPs = [data['test_mAP'] for data in results.values() if data['test_mAP'] is not None]
    
    df = pd.DataFrame({'Augmentation': aug_names, 'test_mAP': test_mAPs})
    df = df.sort_values('test_mAP', ascending=False).reset_index(drop=True)
    
    df['Relative Performance'] = (df['test_mAP'] - df['test_mAP'].min()) / (df['test_mAP'].max() - df['test_mAP'].min())
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(df['Augmentation'], df['Relative Performance'])
    plt.title('Relative Average Test mAP Performance')
    plt.xlabel('Augmentation')
    plt.ylabel('Relative Performance')
    plt.xticks(rotation=90)
    
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f'{df["test_mAP"].iloc[i]:.4f}', 
                 ha='center', va='bottom', rotation=90)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_mAP_relative_performance.png')
    plt.close()
    
    df.to_csv(output_dir / 'test_mAP_rankings.csv', index=False)

def main():
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    base_dir = current_dir / 'local' / 'output_cls'
    result_dir = current_dir / 'result_v2' / 'cls'
    seed_dirs = ['seed2024', 'seed2025', 'seed2026']
    
    result_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = process_directory(base_dir, seed_dirs)
    avg_results = average_results(all_results)
    
    visualize_val_mAP(avg_results, result_dir)
    visualize_test_mAP(avg_results, result_dir)

if __name__ == '__main__':
    main()