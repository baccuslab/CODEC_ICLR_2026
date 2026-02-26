
import scipy
import bscope.ic as bic
import bscope
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.patches import Patch

from skkm import Figaro

F = Figaro(fig_dir='hyperparam', base_path='/home/zalaoui/higanbana/svgs')




top_1 ='/mnt/data/codec/h5s/act_normgrad_top_1_False_resnet50/data.h5'
contrastive = '/mnt/data/codec/h5s/act_normgrad_contrastive_top2_False_resnet50/data.h5'

contrastive_hoyers_all_layers = []
top1_hoyers_all_layers = []


top1_hoyers = []
contrastive_hoyers = []

top1_hoyers_sem = []
contrastive_hoyers_sem = []

datatype1 = 'contributions'
datatype2 = 'contributions'

for l in range(16):
    print(f'Layer {l}')
    top1_data = bic.load_contribution_data(top_1, datatype1, l, 'positive')[0]
    contrastive_data = bic.load_contribution_data(contrastive, datatype2, l, 'positive')[0]

    top1_h=bscope.hoyer(top1_data)
    contrastive_h = bscope.hoyer(contrastive_data)

    top1_hoyers_all_layers.append(top1_h)
    contrastive_hoyers_all_layers.append(contrastive_h)

 
    top1_hoyers.append(np.nanmean(top1_h))
    contrastive_hoyers.append(np.nanmean(contrastive_h))

    top1_hoyers_sem.append(np.nanstd(top1_h))
    contrastive_hoyers_sem.append(np.nanstd(contrastive_h))



plt.figure(figsize=(10,6))
plt.boxplot(contrastive_hoyers_all_layers, positions=np.arange(16)-0.1, showfliers=False, boxprops=dict(color='k'), medianprops=dict(color='k'), widths=0.18)
plt.boxplot(top1_hoyers_all_layers, positions=np.arange(16)+0.1, showfliers=False, boxprops=dict(color='b'), medianprops=dict(color='b'), widths=0.18)
plt.xlim(-1, 16)
plt.ylim(0, 1)
plt.xticks([0, 4, 8, 12, 15], ['0', '4', '8', '12', '15'])
plt.xlabel('Layer')
plt.ylabel('Sparsity (Hoyer)')
plt.title('Sparsity per Layer ')
legend_handles = [
    Patch(facecolor='white', edgecolor='k', label=f'Contrastive Contributions'),
    Patch(facecolor='white', edgecolor='b', label='Top-1 Contributions')
]
plt.legend(handles=legend_handles)
bscope.style_plot(plt.gca())
plt.tight_layout()
F.XX(1.0,1.0)
F.save(f'sparsity_comparision_boxplot_top1_contrastive')
