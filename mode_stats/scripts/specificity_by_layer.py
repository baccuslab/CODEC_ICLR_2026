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

F = fycus.Fycus('specificity_by_layer')
device = 'cuda:0'

# =============================================================================
# SPECIFICITY METRICS
# =============================================================================
# All metrics operate on a single row of the correlation matrix
# (one mode's correlations across all ImageNet classes).
# Higher return value = more specific (correlated with fewer classes).

def specificity_entropy(row):
    """
    Normalized entropy specificity: 1 - H / H_max.
    Uses softmax-style normalization so that the distribution sums to 1
    while preserving relative magnitudes.  H_max = log(num_classes).
    Returns 1 for a delta distribution, 0 for a uniform distribution.
    """
    r = np.clip(row, 0, None)           # ensure non-negative
    total = r.sum()
    if total <= 0:
        return 0.0
    p = r / total
    # avoid log(0)
    mask = p > 0
    h = -np.sum(p[mask] * np.log(p[mask]))
    h_max = np.log(len(row))
    return float(1.0 - h / h_max)


def specificity_gini(row):
    """
    Gini coefficient of the correlation values.
    0 = perfectly uniform (not specific), 1 = all mass on one class (very specific).
    """
    r = np.clip(row, 0, None)
    r_sorted = np.sort(r)               # ascending
    n = len(r_sorted)
    total = r_sorted.sum()
    if total <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.dot(index, r_sorted) - (n + 1) * total) / (n * total))


def specificity_top1_fraction(row):
    """
    Fraction of total correlation mass in the single top class.
    0 = uniform, 1 = all in one class.
    """
    r = np.clip(row, 0, None)
    total = r.sum()
    if total <= 0:
        return 0.0
    return float(r.max() / total)


def specificity_top5_fraction(row):
    """
    Fraction of total correlation mass in the top-5 classes.
    """
    r = np.clip(row, 0, None)
    total = r.sum()
    if total <= 0:
        return 0.0
    return float(np.sort(r)[-5:].sum() / total)


METRICS = {
    'Entropy specificity\n(1 - H/H_max)': specificity_entropy,
    'Gini coefficient': specificity_gini,
    'Top-1 fraction': specificity_top1_fraction,
    'Top-5 fraction': specificity_top5_fraction,
}

# =============================================================================
# MAIN
# =============================================================================

all_results = {}   # keyed by datatype, used for the combined comparison plot

for datatype in ['activations', 'contributions']:
    mode_summary_path = f'/data/h5s/act_normgrad_top_1_False_resnet50/saes/aleph_{datatype}_positive/mode_summary.h5'
    mode_summary = bic.ModeSummary(mode_summary_path)

    layer_indices = list(range(len(mode_summary.layers)))   # 0 … 15

    # For each metric, collect mean and std across modes per layer
    results = {name: {'mean': [], 'std': [], 'median': [], 'p25': [], 'sem': [], 'p75': []}
               for name in METRICS}

    for layer_idx, layer in enumerate(tqdm.tqdm(mode_summary.layers, desc=f'{datatype} layers')):
        corr = layer.imgnet_corr_mtx          # [num_modes, num_classes]

        for metric_name, metric_fn in METRICS.items():
            scores = np.array([metric_fn(corr[i]) for i in range(corr.shape[0])])
            results[metric_name]['mean'].append(scores.mean())
            results[metric_name]['std'].append(scores.std())
            results[metric_name]['sem'].append(scores.std() / np.sqrt(len(scores)))
            results[metric_name]['median'].append(np.median(scores))
            results[metric_name]['p25'].append(np.percentile(scores, 25))
            results[metric_name]['p75'].append(np.percentile(scores, 75))

    # -------------------------------------------------------------------------
    # Plot 1: all four metrics in one figure (mean ± std)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(len(METRICS), 1, sharex=True)
    x = np.array(layer_indices)

    for ax, (metric_name, res) in zip(axes, results.items()):
        mean = np.array(res['mean'])
        std = np.array(res['std'])
        sem = np.array(res['sem'])
        ax.fill_between(x, mean - sem, mean + sem, alpha=0.25)
        ax.plot(x, mean, marker='o', ms=4, lw=1.5)
        ax.set_ylabel(metric_name, fontsize=7)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.3)

    axes[-1].set_xlabel('Layer index')
    axes[-1].set_xticks(x)
    fig.suptitle(f'Mode specificity vs. layer depth ({datatype})', fontsize=9)
    fig.tight_layout()
    F.XX(0.6, 2.5)
    F.save(f'specificity_all_metrics_{datatype}')
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 2: median + IQR band (more robust to outliers)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(len(METRICS), 1, sharex=True)

    for ax, (metric_name, res) in zip(axes, results.items()):
        median = np.array(res['median'])
        p25 = np.array(res['p25'])
        p75 = np.array(res['p75'])
        ax.fill_between(x, p25, p75, alpha=0.30)
        ax.plot(x, median, marker='o', ms=4, lw=1.5)
        ax.set_ylabel(metric_name, fontsize=7)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.3)

    axes[-1].set_xlabel('Layer index')
    axes[-1].set_xticks(x)
    fig.suptitle(f'Mode specificity (median + IQR) vs. layer depth ({datatype})', fontsize=9)
    fig.tight_layout()
    F.XX(0.6, 2.5)
    F.save(f'specificity_median_iqr_{datatype}')
    plt.close()

    # -------------------------------------------------------------------------
    # Plot 3: entropy specificity only, larger panel for clarity
    # -------------------------------------------------------------------------
    key = 'Entropy specificity\n(1 - H/H_max)'
    res = results[key]
    mean = np.array(res['mean'])
    std = np.array(res['std'])
    median = np.array(res['median'])
    p25 = np.array(res['p25'])
    p75 = np.array(res['p75'])

    fig, ax = plt.subplots()
    ax.fill_between(x, p25, p75, alpha=0.25, label='IQR')
    ax.fill_between(x, mean - std, mean + std, alpha=0.15, label='mean ± std')
    ax.plot(x, median, marker='o', ms=5, lw=2, label='Median')
    ax.plot(x, mean, marker='s', ms=4, lw=1.5, ls='--', label='Mean')
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Entropy specificity (1 − H/H_max)')
    ax.set_title(f'Entropy-based mode specificity vs. layer ({datatype})')
    ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.grid(axis='y', alpha=0.3)
    F.XX(0.6, 0.6)
    F.save(f'specificity_entropy_{datatype}')
    plt.close()

    all_results[datatype] = results

# =============================================================================
# Plot 4: activations vs contributions — entropy specificity (mean ± SEM)
# =============================================================================
COLORS = {'activations': 'tab:blue', 'contributions': 'black'}
entropy_key = 'Entropy specificity\n(1 - H/H_max)'

fig, ax = plt.subplots()
x = np.arange(16)

for datatype, color in COLORS.items():
    res = all_results[datatype][entropy_key]
    mean = np.array(res['mean'])
    sem = np.array(res['sem'])
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.35)
    ax.plot(x, mean, color=color, marker='o', ms=4, lw=1.5, label=datatype)

ax.set_xlabel('Layer index')
ax.set_ylabel('Entropy specificity (1 − H/H_max)')
ax.set_title('Activations vs. contributions: entropy specificity by layer')
ax.legend(fontsize=8)
ax.set_xticks(x)
ax.set_ylim(bottom=0)
ax.grid(axis='y', alpha=0.3)
F.XX(0.6, 0.6)
F.save('specificity_comparison')
plt.close()
