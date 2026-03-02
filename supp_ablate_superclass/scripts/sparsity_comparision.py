
import scipy
import bscope.ic as bic
import bscope
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.patches import Patch

from fycus import Fycus

F = Fycus('hyperparam', base_path='/home/zalaoui/higanbana/svgs')




top_1 ='/mnt/data/codec/h5s/act_normgrad_top_1_False_resnet50/data.h5'
contrastive = '/mnt/data/codec/h5s/act_normgrad_contrastive_top2_False_resnet50/data.h5'

contrastive_hoyers_all_layers = []
top1_hoyers_all_layers = []
activations_hoyers_all_layers = []

for l in range(16):
    print(f'Layer {l}')
    top1_data = bic.load_contribution_data(top_1, 'contributions', l, 'positive')[0]
    contrastive_data = bic.load_contribution_data(contrastive, 'contributions', l, 'positive')[0]
    activations_data = bic.load_contribution_data(contrastive, 'activations', l, 'positive')[0]

    top1_hoyers_all_layers.append(bscope.hoyer(top1_data))
    contrastive_hoyers_all_layers.append(bscope.hoyer(contrastive_data))
    activations_hoyers_all_layers.append(bscope.hoyer(activations_data))


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

box_kw = dict(showfliers=False, widths=0.18)

for ax, (blue_data, blue_label) in zip(axes, [
    (top1_hoyers_all_layers, 'Top-1 Contributions'),
    (activations_hoyers_all_layers, 'Activations'),
]):
    ax.boxplot(contrastive_hoyers_all_layers, positions=np.arange(16) - 0.1,
               boxprops=dict(color='k'), medianprops=dict(color='k'),
               whiskerprops=dict(color='k'), capprops=dict(color='k'), **box_kw)
    ax.boxplot(blue_data, positions=np.arange(16) + 0.1,
               boxprops=dict(color='b'), medianprops=dict(color='b'),
               whiskerprops=dict(color='b'), capprops=dict(color='b'), **box_kw)
    ax.set_xlim(-1, 16)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 4, 8, 12, 15])
    ax.set_xticklabels(['0', '4', '8', '12', '15'], fontsize=14)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('Layer', fontsize=20)
    ax.set_ylabel('Sparsity (Hoyer)', fontsize=20)
    ax.set_title('Sparsity per Layer')
    legend_handles = [
        Patch(facecolor='white', edgecolor='k', label='Contrastive Contributions'),
        Patch(facecolor='white', edgecolor='b', label=blue_label),
    ]
    ax.legend(handles=legend_handles, fontsize=10)
    bscope.style_plot(ax)

plt.tight_layout()
F.XX(1.0,1.5) 
F.save('sparsity_comparision_combined_boxplot_ticks')