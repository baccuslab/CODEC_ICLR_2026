import numpy as np
import h5py as h5
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import bscope
from fycus import Fycus


F = Fycus('hyperparam', base_path='/home/zalaoui/higanbana/STSAE')




def extract_metrics(m1_path, threshold=0.2):

    results = {}
    with h5.File(m1_path, 'r') as f:
        for layer_key in f['layers'].keys():
            corr_mtx = f['layers'][layer_key]['imgnet_corr_mtx'][:]
            r2 = f['layers'][layer_key].attrs['r2']
            
            high_corrs = corr_mtx > threshold
            high_features = np.sum(np.sum(high_corrs, axis=0) > 0)
            high_concepts = np.sum(np.sum(high_corrs, axis=1) > 0)

            
            results[int(layer_key)] = {
                'r2': float(r2),
                'high_features': int(high_features),
                'high_concepts': int(high_concepts),
            }
    return results

def run_metrics_extraction(sweep_bases):

    records = []
    
    for sweep_name, sweep_base_dir in sweep_bases.items():
        config_dirs = glob.glob(os.path.join(sweep_base_dir, 'hypersweep_*'))
        print(f"\n=== Sweep: {sweep_name} ({len(config_dirs)} configs) ===")
        
        for config_dir in config_dirs:
            config_name = os.path.basename(config_dir)
            m1_path = os.path.join(config_dir, 'mode_summary.h5')
            
            if not os.path.exists(m1_path):
                print(f"  Skipping {config_name} - no mode_summary.h5")
                continue
            
            print(f"  Extracting: {config_name}")
            params = bscope.parse_config(config_name)
            
            metrics = extract_metrics(m1_path)

            
            for layer_idx, m in metrics.items():
                record = params.copy()
                record['layer'] = layer_idx
                record['sweep'] = sweep_name
                record['r2'] = m['r2']
                record['high_features'] = m['high_features']
                record['high_concepts'] = m['high_concepts']
                records.append(record)
    
    df = pd.DataFrame(records)  
    
    return df



def plot_metrics_figure(df):

    df_grid = df[df['sweep'].isin(['1001', '2002', '483'])].copy()
    df_grid['seed'] = df_grid['sweep'].astype(int)
    df_mlp = df[df['sweep'] == 'mlp_size'].copy()
    
    
    metrics = ['r2', 'high_features', 'high_concepts']
    metric_labels = ['R² Score', 'Modes with Corr > .2', 'Classes with Corr >.2']
    
    panels = [
        {'df': df_grid, 'col': 'atom_l1', 'title': 'L1 Strength', 'fmt': lambda x: f'{x:.0e}' if x > 0 else '0'},
        {'df': df_grid, 'col': 'threshold', 'title': 'Threshold', 'fmt': lambda x: f'{x:.1f}'},
        {'df': df_grid, 'col': 'N', 'title': 'Dict Size (N)', 'fmt': lambda x: f'{int(x)}'},
        {'df': df_grid, 'col': 'seed', 'title': 'Random Seed', 'fmt': lambda x: f'{int(x)}'},
        {'df': df_mlp, 'col': 'mlp_size', 'title': 'MLP Size', 'fmt': lambda x: f'{int(x)}'},
        {'df': df_mlp, 'col': 'nonneg', 'title': 'Non-negative', 'fmt': lambda x: str(x)},
    ]
    
    fig, axes = plt.subplots(3, 6, figsize=(24, 10), sharex='col', sharey='row')
    colors = plt.cm.tab10.colors
    
    for row, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        for col, p in enumerate(panels):
            ax = axes[row, col]
            panel_df = p['df']
            hyperparam_col = p['col']
            
            if len(panel_df) == 0:
                ax.set_title(p['title'] if row == 0 else '')
                ax.set_visible(False)
                continue


            panel_df_filtered = panel_df
            
            grouped = panel_df_filtered.groupby([hyperparam_col, 'layer'])[metric].agg(['mean', 'std', 'count']).reset_index()
            grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])


            unique_vals = sorted(grouped[hyperparam_col].unique())
            
            for i, val in enumerate(unique_vals):
                subset = grouped[grouped[hyperparam_col] == val].sort_values('layer')
                layers = subset['layer'].values
                means = subset['mean'].values
                sems = subset['sem'].values
                
                label = p['fmt'](val)
                color = colors[i % len(colors)]
                
                ax.plot(layers, means, 'o-', color=color, label=label, markersize=4)
                ax.fill_between(layers, means - sems, means + sems, color=color, alpha=0.2)
            
            if row == 0:
                ax.set_title(p['title'])
                ax.set_ylim([0, 1] if metric == 'r2' else [0, None])
            if row == 2:
                ax.set_xlabel('Layer')
            if col == 0:
                ax.set_ylabel(metric_label)
                
            ax.legend(fontsize=7, loc='best')
    
    fig.suptitle('SAE Metrics Across Hyperparameters', fontsize=14, y=1.01)
    plt.tight_layout()
    
    # F.XX(1.0, 1.5)
    # F.save(f'reconstruction_metrics_baseline_vs_sweep')
    print("Figure generated.")
    plt.show()
    return fig


if __name__ == '__main__':
    SWEEP_BASES = {
        '1001': '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/sweep_int_grad_top_1_False_resnet50_steps_10_1001_positive_STSAE',
        '2002': '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/sweep_int_grad_top_1_False_resnet50_steps_10_2002_positive_STSAE',
        '483': '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/sweep_int_grad_top_1_False_resnet50_steps_10_483_positive_STSAE',
        'mlp_size': '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/sweep_int_grad_top_1_False_resnet50_steps_10_483_mlpsize_nonneg_STSAE'
    }
    OUTPUT_DIR = '/home/zalaoui/codec_zaki/results'
    
    df = run_metrics_extraction(SWEEP_BASES)
    plot_metrics_figure(df)