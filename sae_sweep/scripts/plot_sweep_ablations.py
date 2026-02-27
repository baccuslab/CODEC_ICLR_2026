import pickle
from tkinter import Frame
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob
import bscope
from fycus import Fycus


RESULTS_DIR = os.path.expanduser('~/codec_zaki/iclr_scripts/STSAE_ABLATIONS_MODEFIX')
F = Fycus('hyperparam', base_path='/home/zalaoui/higanbana/STSAE')
colors = plt.cm.tab10.colors

TOPK = 'top1'
LAYERS = [3, 7, 13, 15]
PCTS = [25, 50]


# ============================================================================
# LOAD ALL RESULTS
# ============================================================================
all_results = {}
unique_files = set()


for filepath in glob.glob(os.path.join(RESULTS_DIR, '*.pkl')):


    unique_files.add(filepath)
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    unique_key = f"{data['sweep_name']}_{data['config_name']}"
    all_results[unique_key] = data
    all_results[data['config_name']] = data

n_trials = np.array(list(all_results.values())[0]['aleph'][LAYERS[0]][PCTS[0]][f'og_{TOPK}']).shape[0]
print(f"Loaded {len(unique_files)} unique result files ({len(all_results)} total dictionary entries)")

# ============================================================================
# HELPER: Calculate AUC difference
# ============================================================================
def calc_auc_diff(results, source='aleph'):
    """Calculate AUC difference: (targ_auc - offtarg_auc) / offtarg_auc per layer"""
    auc_diffs = {li: {'mean': None, 'sem': None, 'values': []} for li in LAYERS}
    
    for li in LAYERS:
        og_data_25 = np.array(results[source][li][25][f'og_{TOPK}'])
        pert_data_25 = np.array(results[source][li][25][f'pert_{TOPK}'])
        og_data_50 = np.array(results[source][li][50][f'og_{TOPK}'])
        pert_data_50 = np.array(results[source][li][50][f'pert_{TOPK}'])
        
        trial_auc_diffs = []
        
        for trial in range(n_trials):
            if og_data_25.ndim == 1:
                continue
            
            targ_ratio_25 = pert_data_25[trial, 0] / og_data_25[trial, 0] if og_data_25[trial, 0] > 0 else np.nan
            targ_ratio_50 = pert_data_50[trial, 0] / og_data_50[trial, 0] if og_data_50[trial, 0] > 0 else np.nan
            
            offtarg_ratio_25 = pert_data_25[trial, 1] / og_data_25[trial, 1] if og_data_25[trial, 1] > 0 else np.nan
            offtarg_ratio_50 = pert_data_50[trial, 1] / og_data_50[trial, 1] if og_data_50[trial, 1] > 0 else np.nan
            
            targ_auc = bscope.compute_auc(PCTS, [targ_ratio_25, targ_ratio_50])
            offtarg_auc = bscope.compute_auc(PCTS, [offtarg_ratio_25, offtarg_ratio_50])
            
            if offtarg_auc != 0 and not np.isnan(offtarg_auc):
                auc_diff = (targ_auc - offtarg_auc) / offtarg_auc
                trial_auc_diffs.append(auc_diff)
        
        auc_diffs[li]['values'] = np.array(trial_auc_diffs)
        auc_diffs[li]['mean'] = np.nanmean(trial_auc_diffs)
        auc_diffs[li]['sem'] = stats.sem(trial_auc_diffs, nan_policy='omit')
    
    return auc_diffs

# ============================================================================
# CREATE COMBINED FIGURE WITH ALL PLOTS
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
axes = axes.flatten()

# # ============================================================================
# # PLOT 1: Dictionary Size Multiplier
# # ============================================================================
ax = axes[0]
bscope.style_plot(ax)

dict_size_configs = {
    'N=1': 'hypersweep_0.7_1_0.0001',
    'N=3': 'hypersweep_0.7_3_0.0001',
    'N=5': 'hypersweep_0.7_5_0.0001',
}



for i, (label, config_name) in enumerate(dict_size_configs.items()):
    if config_name not in all_results:
        continue
    results = all_results[config_name]
    auc_diffs = calc_auc_diff(results, 'sweep')
    means = [auc_diffs[li]['mean'] for li in LAYERS]
    sems = [auc_diffs[li]['sem'] for li in LAYERS]
    
    color = colors[i % len(colors)]
    ax.plot(LAYERS, means, 'o-', color=color, label=label, linewidth=2, markersize=4)
    ax.fill_between(LAYERS, 
                     np.array(means) - np.array(sems), 
                     np.array(means) + np.array(sems), 
                     color=color, alpha=0.2)

ax.set_title('Dictionary Size Multiplier')
ax.legend(frameon=False, fontsize=8)
ax.set_ylim(-1.1, 1.4)
# ============================================================================
# PLOT 2: Threshold
# ============================================================================
# PLOT 2: Threshold
ax = axes[1]
bscope.style_plot(ax)

threshold_configs = {
    'Threshold=0.5': 'hypersweep_0.5_5_0.0001',
    'Threshold=0.7': 'hypersweep_0.7_5_0.0001',
    'Threshold=0.9': 'hypersweep_0.9_5_0.0001',
}

# Just loop through all three normally
for i, (label, config_name) in enumerate(threshold_configs.items()):
    if config_name not in all_results:
        continue
    results = all_results[config_name]
    auc_diffs = calc_auc_diff(results, 'sweep')  # Use 'sweep' for all
    means = [auc_diffs[li]['mean'] for li in LAYERS]
    sems = [auc_diffs[li]['sem'] for li in LAYERS]
    
    color = colors[i % len(colors)]
    ax.plot(LAYERS, means, 'o-', color=color, label=label, linewidth=2, markersize=4)
    ax.fill_between(LAYERS, 
                     np.array(means) - np.array(sems), 
                     np.array(means) + np.array(sems), 
                     color=color, alpha=0.2)

ax.set_title('Threshold')
ax.legend(frameon=False, fontsize=8)
ax.set_ylim(-1.1, 1.4)
# ============================================================================
# PLOT 3: Atom L1
# ============================================================================
ax = axes[2]
bscope.style_plot(ax)

atom_l1_configs = {
    'L1=0': 'hypersweep_0.9_5_0',
    'L1=1e-4': 'hypersweep_0.9_5_0.0001', 
    'L1=1e-2': 'hypersweep_0.9_5_0.01',    
}



for i, (label, config_name) in enumerate(atom_l1_configs.items()):
    if config_name not in all_results:
        continue
    results = all_results[config_name]
    auc_diffs = calc_auc_diff(results, 'sweep')
    means = [auc_diffs[li]['mean'] for li in LAYERS]
    sems = [auc_diffs[li]['sem'] for li in LAYERS]
    
    color = colors[i % len(colors)]
    ax.plot(LAYERS, means, 'o-', color=color, label=label, linewidth=2, markersize=4)
    ax.fill_between(LAYERS, 
                     np.array(means) - np.array(sems), 
                     np.array(means) + np.array(sems), 
                     color=color, alpha=0.2)

ax.set_title('Atom L1 Regularization')
ax.legend(frameon=False, fontsize=8)
ax.set_ylim(-1.1, 1.4)
# ============================================================================
# PLOT 4: Seed
# ============================================================================
ax = axes[3]
bscope.style_plot(ax)

# ============================================================================
# PLOT 4: Seed
# ============================================================================
ax = axes[3]
bscope.style_plot(ax)

seed_configs = {
    'Seed=483': ('sweep_int_grad_top_1_False_resnet50_steps_10_483_positive_STSAE', 'hypersweep_0.9_5_0.0001'),
    'Seed=1001': ('sweep_int_grad_top_1_False_resnet50_steps_10_1001_positive_STSAE', 'hypersweep_0.9_5_0.0001'),
    'Seed=2002': ('sweep_int_grad_top_1_False_resnet50_steps_10_2002_positive_STSAE', 'hypersweep_0.9_5_0.0001'),
}

i = 0
for label, (sweep_name, config_name) in seed_configs.items():
    for key, result_data in all_results.items():
        if result_data.get('sweep_name') == sweep_name and result_data.get('config_name') == config_name:
            auc_diffs = calc_auc_diff(result_data, 'sweep')
            means = [auc_diffs[li]['mean'] for li in LAYERS]
            sems = [auc_diffs[li]['sem'] for li in LAYERS]
            
            color = colors[i % len(colors)]
            ax.plot(LAYERS, means, 'o-', color=color, label=label, linewidth=2, markersize=4)
            ax.fill_between(LAYERS, 
                           np.array(means) - np.array(sems), 
                           np.array(means) + np.array(sems), 
                           color=color, alpha=0.2)
            i += 1
            break

ax.set_title('Seed')
ax.legend(frameon=False, fontsize=8)
ax.set_ylim(-1.1, 1.4)
# # ============================================================================
# # PLOT 5: MLP Size
# # ============================================================================
ax = axes[4]
bscope.style_plot(ax)

mlp_size_configs = {
    'MLP=128': 'hypersweep_mlpsize_128_nonneg_False_0.5_5_0.0001',
    'MLP=512': 'hypersweep_mlpsize_512_nonneg_False_0.5_5_0.0001',
    'MLP=2048': 'hypersweep_mlpsize_2048_nonneg_False_0.5_5_0.0001',
    'MLP=4096': 'hypersweep_mlpsize_4096_nonneg_False_0.5_5_0.0001',
}

for i, (label, config_name) in enumerate(mlp_size_configs.items()):
    if config_name not in all_results:
        continue
    results = all_results[config_name]
    auc_diffs = calc_auc_diff(results, 'sweep')
    means = [auc_diffs[li]['mean'] for li in LAYERS]
    sems = [auc_diffs[li]['sem'] for li in LAYERS]
    
    color = colors[i % len(colors)]
    ax.plot(LAYERS, means, 'o-', color=color, label=label, linewidth=2, markersize=4)
    ax.fill_between(LAYERS, 
                     np.array(means) - np.array(sems), 
                     np.array(means) + np.array(sems), 
                     color=color, alpha=0.2)

ax.set_title('MLP Size')
ax.legend(frameon=False, fontsize=8)
ax.set_ylim(-1.1, 1.4)


plt.subplots_adjust(hspace=0.35)

for i in [3,4,5]:
    axes[i].set_xlabel('Layer')
for i in [0,3]:
    axes[i].set_ylabel('AUC')
axes[5].axis('off')  # Add this at the end before plt.show()
plt.tight_layout()


# Uncomment to save:
# F.XX(1.0,1.5) 
# F.save('ablations_comparison_final_arxiv_STSAE_TOPMODE')

plt.show()