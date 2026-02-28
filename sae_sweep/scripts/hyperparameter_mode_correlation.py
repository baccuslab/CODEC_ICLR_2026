import numpy as np
import h5py as h5
import glob
import os
import pandas as pd
from fycus import Fycus
from scipy import stats
import matplotlib.pyplot as plt
import bscope


F = Fycus('hyperparam', base_path='/home/zalaoui/higanbana/STSAE')


def compare_mode_summaries(m0_path, m1_path, corr_threshold=0.2, mode='sum'):

    with h5.File(m0_path, 'r') as f0, h5.File(m1_path, 'r') as f1:
        layers_0 = set(f0['layers'].keys())
        layers_1 = set(f1['layers'].keys())
        common_layers = sorted([int(l) for l in layers_0 & layers_1])
        
        results = {}
        
        for layer_idx in common_layers:
            layer_key = str(layer_idx)
            
            corr_0 = f0['layers'][layer_key]['imgnet_corr_mtx'][:]
            corr_1 = f1['layers'][layer_key]['imgnet_corr_mtx'][:]
            dict_0 = f0['layers'][layer_key]['dictionary'][:]
            dict_1 = f1['layers'][layer_key]['dictionary'][:]
            
            n_classes = corr_0.shape[0]
            corrs = []
            n_modes_m0 = []
            n_modes_m1 = []
            
            for class_idx in range(n_classes):
                if mode == 'top':
                    top_idx_0 = np.argmax(corr_0[class_idx, :])
                    top_idx_1 = np.argmax(corr_1[class_idx, :])
                    v0 = dict_0[top_idx_0]
                    v1 = dict_1[top_idx_1]
                    n_modes_m0.append(1)
                    n_modes_m1.append(1)
                else:  
                    modes_0 = np.where(corr_0[class_idx, :] > corr_threshold)[0]
                    modes_1 = np.where(corr_1[class_idx, :] > corr_threshold)[0]
                    n_modes_m0.append(len(modes_0))
                    n_modes_m1.append(len(modes_1))
                    
                    if len(modes_0) == 0:
                        modes_0 = np.array([np.argmax(corr_0[class_idx, :])])
                        n_modes_m0[-1] = 1  
                    if len(modes_1) == 0:
                        modes_1 = np.array([np.argmax(corr_1[class_idx, :])])
                        n_modes_m1[-1] = 1 
                        
                    
                    v0 = dict_0[modes_0].sum(axis=0)
                    v1 = dict_1[modes_1].sum(axis=0)
                
                corr = np.corrcoef(v0, v1)[0, 1]
                corrs.append(corr)
            
            results[layer_idx] = {
                'corrs': np.array(corrs),
                'n_modes_m0': np.array(n_modes_m0),
                'n_modes_m1': np.array(n_modes_m1),
            }
        
        return results

def run_comparison(baseline_path, sweep_bases, output_dir=None, corr_threshold=0.2, mode='sum'):

    records = []
    for sweep_name, sweep_base_dir in sweep_bases.items():  
        config_dirs = glob.glob(os.path.join(sweep_base_dir, 'hypersweep_*'))
        print(f"\n=== Sweep: {sweep_name} ({len(config_dirs)} configs) ===")
        
        print(f"Found {len(config_dirs)} configs to compare against baseline")
        print(f"Base path: {baseline_path}")
        print(f"Correlation threshold: {corr_threshold}")
        
        for config_dir in config_dirs:
            config_name = os.path.basename(config_dir)
            m1_path = os.path.join(config_dir, 'mode_summary.h5')
            
            if not os.path.exists(m1_path):
                print(f"  Skipping {config_name} - no mode_summary.h5")
                continue
            
            print(f"  Comparing: {config_name}")
            params = bscope.parse_config(config_name)
            

            layer_results = compare_mode_summaries(baseline_path, m1_path, corr_threshold, mode=mode)

            
            for layer_idx, res in layer_results.items():
                corrs = res['corrs']
                n_modes_m0 = res['n_modes_m0']
                n_modes_m1 = res['n_modes_m1']
                
                record = params.copy()
                record['layer'] = layer_idx
                record['sweep'] = sweep_name  
                record['mean_corr'] = np.nanmean(corrs)
                record['std_corr'] = np.nanstd(corrs)
                record['median_corr'] = np.nanmedian(corrs)
                record['n_valid_classes'] = np.sum(~np.isnan(corrs))
                record['mean_n_modes_m0'] = np.mean(n_modes_m0)
                record['mean_n_modes_m1'] = np.mean(n_modes_m1)
                record['classes_with_0_modes_m0'] = np.sum(n_modes_m0 == 0)
                record['classes_with_0_modes_m1'] = np.sum(n_modes_m1 == 0)
                records.append(record)
        
    df = pd.DataFrame(records)
    return df

def plot_correlation_figure(df, mode, output_path=None):

    # Split by sweep type
    df_grid = df[df['sweep'].isin(['1001', '2002', '483'])].copy()
    df_grid['seed'] = df_grid['sweep'].astype(int)  
    df_mlp = df[df['sweep'] == 'mlp_size'].copy()
    
    # Define panels
    panels = [
        {'df': df_grid, 'col': 'atom_l1', 'title': 'L1 Strength', 'fmt': lambda x: f'{x:.0e}' if x > 0 else '0'},
        {'df': df_grid, 'col': 'threshold', 'title': 'Threshold', 'fmt': lambda x: f'{x:.1f}'},
        {'df': df_grid, 'col': 'N', 'title': 'Dict Size (N)', 'fmt': lambda x: f'{int(x)}'},
        {'df': df_grid, 'col': 'seed', 'title': 'Random Seed', 'fmt': lambda x: f'{int(x)}'}, 
        {'df': df_mlp, 'col': 'mlp_size', 'title': 'MLP Size', 'fmt': lambda x: f'{int(x)}'},
        {'df': df_mlp, 'col': 'nonneg', 'title': 'Non-negative', 'fmt': lambda x: str(x)},
    ]
    
    fig, axes = plt.subplots(1, 6, figsize=(18, 4), sharey=True)
    colors = plt.cm.tab10.colors
        
    for idx, p in enumerate(panels):
        ax = axes[idx]
        panel_df = p['df']
        col = p['col']

        grouped = panel_df.groupby([col, 'layer'])['mean_corr'].agg(['mean', 'std', 'count']).reset_index()
        grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
        

        unique_vals = sorted(grouped[col].unique())
        
        for i, val in enumerate(unique_vals):
            subset = grouped[grouped[col] == val].sort_values('layer')
            layers = subset['layer'].values
            means = subset['mean'].values

            
            label = p['fmt'](val)
            color = colors[i % len(colors)]
            
            ax.plot(layers, means, 'o-', color=color, label=label, markersize=4)
            sems = subset['sem'].values
            ax.fill_between(layers, means - sems, means + sems, color=color, alpha=0.2)
            ax.set_ylim(0, 1)
        
        ax.set_xlabel('Layer')
        ax.set_title(p['title'])
        ax.legend(fontsize=8, loc='upper right')


    axes[0].set_ylabel('Correlation Coefficient to Baseline SAE ')
    
    fig.suptitle(f'Mode Stability Across SAE Hyperparameters ({mode} mode)', fontsize=12, y=1.02)
    plt.tight_layout()


    # F.XX(1.0,1.5)
    # F.save(f'correlation_baseline_vs_sweep_{mode}')
    plt.show()
    print('FIGURE SAVED VIA FIGARO')
    return fig




BASE_PATH = '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/aleph_contributions_positive/mode_summary.h5'
SWEEP_BASES = {
    '1001': '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/sweep_int_grad_top_1_False_resnet50_steps_10_1001_positive_STSAE',
    '2002': '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/sweep_int_grad_top_1_False_resnet50_steps_10_2002_positive_STSAE',
    '483': '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/sweep_int_grad_top_1_False_resnet50_steps_10_483_positive_STSAE',
    'mlp_size': '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/sweep_int_grad_top_1_False_resnet50_steps_10_483_mlpsize_nonneg_STSAE'
}
OUTPUT_DIR = '/home/zalaoui/codec_zaki/results'
CORR_THRESHOLD = 0.2

for mode in ['sum']:
    df = run_comparison(BASE_PATH, SWEEP_BASES, OUTPUT_DIR, CORR_THRESHOLD, mode=mode)
    output_path = os.path.join(OUTPUT_DIR, f'stability_figure_corr_{mode}.svg')
    plot_correlation_figure(df, mode=mode, output_path=output_path)

