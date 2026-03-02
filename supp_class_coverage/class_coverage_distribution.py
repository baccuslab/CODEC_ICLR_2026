# 1. IMPORTS
# =============================================================================

from IPython import embed

import bopt
import bscope
import bscope.ic as bic
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import tqdm
from scipy.ndimage import median_filter, gaussian_filter

import fycus

F = fycus.Fycus('class_coverage_distribution')
device = 'cuda:0'

layers_to_use = [0, 4, 9, 14]
# Subsample class-rank positions for boxplot (1000 classes is too dense)
# We'll compute percentiles manually for efficiency
BOXPLOT_POSITIONS = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 999])


def percentile_summary(matrix, positions):
    """
    For each class-rank position in `positions`, return percentile stats
    across all modes (rows of matrix).
    matrix: [num_modes, num_classes], already sorted descending per row.
    Returns dict of arrays indexed by position.
    """
    stats = {
        'positions': positions,
        'p10': np.percentile(matrix[:, positions], 10, axis=0),
        'p25': np.percentile(matrix[:, positions], 25, axis=0),
        'p50': np.percentile(matrix[:, positions], 50, axis=0),
        'p75': np.percentile(matrix[:, positions], 75, axis=0),
        'p90': np.percentile(matrix[:, positions], 90, axis=0),
        'mean': matrix[:, positions].mean(axis=0),
    }
    return stats


for datatype in ['activations', 'contributions']:
    mode_summary_path = f'/data/h5s/act_normgrad_top_1_False_resnet50/saes/aleph_{datatype}_positive/mode_summary.h5'

    mode_summary = bic.ModeSummary(mode_summary_path)
    print(mode_summary.layers)

    avgs = []
    sorted_matrices = []

    for layer_idx, layer in enumerate(mode_summary.layers):
        if layer_idx not in layers_to_use:
            continue
        correlation_matrix = layer.imgnet_corr_mtx
        num_classes = correlation_matrix.shape[1]

        sorted_indices = np.argsort(correlation_matrix, axis=1)[:, ::-1]
        sorted_corr = np.take_along_axis(correlation_matrix, sorted_indices, axis=1)

        avgs.append(sorted_corr.mean(axis=0))
        sorted_matrices.append(sorted_corr)

    # -------------------------------------------------------------------------
    # Plot 1: mean line + shaded IQR band (10-90th percentile)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots()
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(layers_to_use)))
    x = np.arange(num_classes)

    for avg, smat, layer, color in zip(avgs, sorted_matrices, layers_to_use, colors):
        p10 = np.percentile(smat, 10, axis=0)
        p25 = np.percentile(smat, 25, axis=0)
        p75 = np.percentile(smat, 75, axis=0)
        p90 = np.percentile(smat, 90, axis=0)
        ax.fill_between(x, p10, p90, alpha=0.15, color=color, label=None)
        ax.fill_between(x, p25, p75, alpha=0.30, color=color, label=None)
        ax.plot(x, avg, color=color, lw=1.5, label=f'Layer {layer}')

    ax.set_xlabel('Class rank (sorted by correlation per mode)')
    ax.set_ylabel('Correlation')
    ax.set_title('Mean + IQR band of sorted correlations')
    ax.legend(fontsize=8)
    F.XX(0.6, 1.5)
    F.save(f'mean_band_{datatype}')
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 2: zoomed version (top-50 ranks)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots()
    for avg, smat, layer, color in zip(avgs, sorted_matrices, layers_to_use, colors):
        p25 = np.percentile(smat, 25, axis=0)
        p75 = np.percentile(smat, 75, axis=0)
        ax.fill_between(x[:50], p25[:50], p75[:50], alpha=0.30, color=color)
        ax.plot(x[:50], avg[:50], color=color, lw=1.5, label=f'Layer {layer}')

    ax.set_xlabel('Class rank (sorted by correlation per mode)')
    ax.set_ylabel('Correlation')
    ax.set_title('Zoomed: mean + IQR of sorted correlations')
    ax.legend(fontsize=8)
    F.XX(0.5, 0.5)
    F.save(f'zoom_mean_band_{datatype}')
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 3: violin plot at sampled class-rank positions
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(len(layers_to_use), 1, sharex=True)
    pos_labels = [str(p) for p in BOXPLOT_POSITIONS]

    for ax, smat, layer in zip(axes, sorted_matrices, layers_to_use):
        data = [smat[:, p] for p in BOXPLOT_POSITIONS]
        parts = ax.violinplot(data, positions=np.arange(len(BOXPLOT_POSITIONS)),
                              showmedians=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_alpha(0.6)
        ax.set_ylabel(f'L{layer}', fontsize=7, rotation=0, labelpad=20)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    axes[-1].set_xticks(np.arange(len(BOXPLOT_POSITIONS)))
    axes[-1].set_xticklabels(pos_labels, fontsize=7)
    axes[-1].set_xlabel('Class rank (log-spaced samples)')
    fig.suptitle(f'Distribution of correlations at sampled ranks ({datatype})', fontsize=9)
    F.XX(0.7, 2.0)
    F.save(f'violin_{datatype}')
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 4: boxplot at sampled class-rank positions
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(len(layers_to_use), 1, sharex=True)

    for ax, smat, layer in zip(axes, sorted_matrices, layers_to_use):
        data = [smat[:, p] for p in BOXPLOT_POSITIONS]
        bp = ax.boxplot(data, positions=np.arange(len(BOXPLOT_POSITIONS)),
                        widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', lw=1.5))
        for patch in bp['boxes']:
            patch.set_alpha(0.6)
        ax.set_ylabel(f'L{layer}', fontsize=7, rotation=0, labelpad=20)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    axes[-1].set_xticks(np.arange(len(BOXPLOT_POSITIONS)))
    axes[-1].set_xticklabels(pos_labels, fontsize=7)
    axes[-1].set_xlabel('Class rank (log-spaced samples)')
    fig.suptitle(f'Boxplot of correlations at sampled ranks ({datatype})', fontsize=9)
    F.XX(0.7, 2.0)
    F.save(f'boxplot_{datatype}')
    plt.close()
