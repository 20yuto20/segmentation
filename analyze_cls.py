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
                if aug_dir.name.startswith('RA1_') and aug_dir.name.endswith('_Randmag'):
                    aug_name = aug_dir.name.split('_')[1]
                else:
                    aug_name = aug_dir.name
                
                csv_path = aug_dir / 'output.csv'
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    results[seed][aug_name] = {
                        'val_mAP': df['val_mAP'].tolist(),
                        'test_mAP': df['test_mAP'].dropna().tolist()[-1] if not df['test_mAP'].dropna().empty else None,
                        'avg_inference_time': df['avg_inference_time'].dropna().tolist()[-1] if not df['avg_inference_time'].dropna().empty else None
                    }
    return results

def average_results(results):
    avg_results = {}
    for aug_name in results[list(results.keys())[0]]:
        val_mAPs = [results[seed][aug_name]['val_mAP'] for seed in results if aug_name in results[seed]]
        test_mAPs = [results[seed][aug_name]['test_mAP'] for seed in results if aug_name in results[seed] and results[seed][aug_name]['test_mAP'] is not None]
        inference_times = [results[seed][aug_name]['avg_inference_time'] for seed in results if aug_name in results[seed] and results[seed][aug_name]['avg_inference_time'] is not None]
        
        min_length = min(len(mAP) for mAP in val_mAPs)
        val_mAPs = [mAP[:min_length] for mAP in val_mAPs]
        
        avg_results[aug_name] = {
            'val_mAP': np.mean(val_mAPs, axis=0).tolist(),
            'test_mAP': np.mean(test_mAPs) if test_mAPs else None,
            'avg_inference_time': np.mean(inference_times) if inference_times else None
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

def visualize_inference_time(results, output_dir):
    aug_names = list(results.keys())
    inference_times = [data['avg_inference_time'] for data in results.values() if 'avg_inference_time' in data]
    
    df = pd.DataFrame({'Augmentation': aug_names, 'avg_inference_time': inference_times})
    df = df.sort_values('avg_inference_time').reset_index(drop=True)
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(df['Augmentation'], df['avg_inference_time'])
    plt.title('Average Inference Time per Image')
    plt.xlabel('Augmentation')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=90)
    
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{df["avg_inference_time"].iloc[i]:.6f}',
                ha='center', va='bottom', rotation=90)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_time_comparison.png')
    plt.close()
    
    df.to_csv(output_dir / 'inference_time_rankings.csv', index=False)

def visualize_performance_vs_time(results, output_dir):
    aug_names = list(results.keys())
    test_mAPs = [data['test_mAP'] for data in results.values() if data['test_mAP'] is not None]
    inference_times = [data['avg_inference_time'] for data in results.values() if 'avg_inference_time' in data]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(inference_times, test_mAPs)
    
    for i, aug_name in enumerate(aug_names):
        plt.annotate(aug_name, (inference_times[i], test_mAPs[i]))
    
    plt.title('Test mAP vs Inference Time Trade-off')
    plt.xlabel('Inference Time (seconds)')
    plt.ylabel('Test mAP')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_vs_time_tradeoff.png')
    plt.close()
    
    df = pd.DataFrame({
        'Augmentation': aug_names,
        'test_mAP': test_mAPs,
        'avg_inference_time': inference_times
    })
    df.to_csv(output_dir / 'performance_time_tradeoff.csv', index=False)

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
    visualize_inference_time(avg_results, result_dir)
    visualize_performance_vs_time(avg_results, result_dir)

if __name__ == '__main__':    main()