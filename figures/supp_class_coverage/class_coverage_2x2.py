# 1. IMPORTS
# =============================================================================

from IPython import embed

import bopt
import bscope
import bscope.ic as bic
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from scipy.ndimage import median_filter, gaussian_filter

import fycus

F = fycus.Fycus('class_coverage_2x2')
F.XX(0.75, .75)
device = 'cuda:0'

bic.get_model('resnet50', imagenet_path='/data/imagenet/', return_layers=True, device=device)

layers_to_use = [2, 6, 12, 15]
N_ZOOM = 60

# 2. COLLECT DATA
# =============================================================================

all_data = {}
for datatype in ['contributions', 'activations']:
    mode_summary_path = f'/data/h5s/int_grad_top_1_False_resnet50_steps_10/saes/aleph_{datatype}_positive/mode_summary.h5'
    mode_summary = bic.ModeSummary(mode_summary_path)
    print(mode_summary.layers)

    class_avgs, class_stds = [], []
    mode_avgs, mode_stds = [], []

    for layer_idx, layer in enumerate(mode_summary.layers):
        if layer_idx not in layers_to_use:
            continue
        correlation_matrix = layer.imgnet_corr_mtx
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)  # Replace NaNs with 0 for visualization
        plt.imshow(correlation_matrix, aspect='auto', cmap='magma', interpolation='none', clim=(0, 0.35))

        plt.colorbar()
        plt.title(f'{datatype} - Layer {layer_idx}')
        F.save(f'{datatype}_layer_{layer_idx}_correlation_matrix')

        num_modes = correlation_matrix.shape[0]
        num_classes = correlation_matrix.shape[1]
        assert num_classes == 1000, f"Expected 1000 classes, got {num_classes}"

        # For each class, sort modes by correlation descending -> (num_classes, num_modes)
        class_coverage = np.array([
            correlation_matrix[np.argsort(correlation_matrix[:, c])[::-1], c]
            for c in range(num_classes)
        ])
        
        class_coverage = np.nan_to_num(class_coverage, nan=0.0)  # Replace NaNs with 0 for visualization
        plt.imshow(class_coverage, aspect='auto', cmap='magma', interpolation='none', clim=(0, 0.35))
        plt.colorbar()
        plt.title(f'{datatype} - Layer {layer_idx}')
        F.save(f'{datatype}_layer_{layer_idx}_class_coverage')

        class_avgs.append(np.nanmean(class_coverage, axis=0))
        class_stds.append(np.nanstd(class_coverage, axis=0))

        # For each mode, sort classes by correlation descending -> (num_modes, num_classes)
        mode_coverage = np.array([
            correlation_matrix[m, np.argsort(correlation_matrix[m, :])[::-1]]
            for m in range(num_modes)
        ])
        mode_avgs.append(np.nanmean(mode_coverage, axis=0))
        mode_stds.append(np.nanstd(mode_coverage, axis=0))

    all_data[datatype] = {
        'class': (class_avgs, class_stds),
        'mode':  (mode_avgs, mode_stds),
    }

# 3. PLOT 2x2 GRID
# =============================================================================
# Rows: top = avg over classes (x=mode rank), bottom = avg over modes (x=class rank)
# Cols: left = contributions, right = activations

fig, axes = plt.subplots(2, 2, figsize=(8, 6))

col_datatypes = ['contributions', 'activations']
col_titles    = ['Contributions', 'Activations']
row_types     = ['class', 'mode']
row_ylabels   = ['Avg over classes\nAvg Correlation', 'Avg over modes\nAvg Correlation']
row_xlabels   = ['Mode rank (top 60)', 'Class rank (top 60)']

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for row_idx, (avg_type, ylabel, xlabel) in enumerate(zip(row_types, row_ylabels, row_xlabels)):
    for col_idx, (datatype, col_title) in enumerate(zip(col_datatypes, col_titles)):
        ax = axes[row_idx, col_idx]
        avgs, sems = all_data[datatype][avg_type]

        for i, (avg, sem, layer) in enumerate(zip(avgs, sems, layers_to_use)):
            y = avg[:N_ZOOM]
            e = sem[:N_ZOOM]
            x = np.arange(len(y))
            color = colors[i % len(colors)]
            ax.errorbar(x, y, yerr=e, label=f'Layer {layer}', color=color,
                        errorevery=5, capsize=2, capthick=0.8, elinewidth=0.8)

        ax.set_ylim(0, 0.75)
        ax.set_xlabel(xlabel)
        if col_idx == 0:
            ax.set_ylabel(ylabel)
        if row_idx == 0:
            ax.set_title(col_title)
        if row_idx == 0 and col_idx == 1:
            ax.legend(loc='upper right')

plt.tight_layout()
F.XX(0.75, .75)
F.save('class_coverage_2x2')

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
for row_idx, (avg_type, ylabel, xlabel) in enumerate(zip(row_types, row_ylabels, row_xlabels)):
    for col_idx, (datatype, col_title) in enumerate(zip(col_datatypes, col_titles)):
        ax = axes[row_idx, col_idx]
        avgs, sems = all_data[datatype][avg_type]

        for i, (avg, sem, layer) in enumerate(zip(avgs, sems, layers_to_use)):
            y = avg
            e = sem
            x = np.arange(len(y))
            color = colors[i % len(colors)]
            ax.errorbar(x, y, yerr=e, label=f'Layer {layer}', color=color,
                        errorevery=5, capsize=2, capthick=0.8, elinewidth=0.8)

        ax.set_ylim(0, 0.75)
        ax.set_xlabel(xlabel)
        if col_idx == 0:
            ax.set_ylabel(ylabel)
        if row_idx == 0:
            ax.set_title(col_title)
        if row_idx == 0 and col_idx == 1:
            ax.legend(loc='upper right')

plt.tight_layout()
F.XX(0.75, .75)
F.save('class_coverage_2x2_zout')
