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

F = fycus.Fycus('class_coverage_final')
device = 'cuda:0'

bic.get_model('resnet50', imagenet_path='/data/imagenet/', return_layers=True, device=device)

for datatype in ['activations', 'contributions']:
    mode_summary_path = f'/data/h5s/act_normgrad_top_1_False_resnet50/saes/aleph_{datatype}_positive/mode_summary.h5'

    mode_summary = bic.ModeSummary(mode_summary_path)
    print(mode_summary.layers)

    avgs = []
    # layers_to_use=[0,4,9,14]
    layers_to_use=[2,6,12,15]
    for layer_idx, layer in enumerate(mode_summary.layers):
        if layer_idx not in layers_to_use:
            continue
        correlation_matrix = layer.imgnet_corr_mtx 
        num_modes = correlation_matrix.shape[0]
        num_classes = correlation_matrix.shape[1]

        # Sort each row 
        sorted_indices = np.argsort(correlation_matrix, axis=1)[:, ::-1]
        sorted_correlation_matrix = np.take_along_axis(correlation_matrix, sorted_indices, axis=1)

        avgs.append(sorted_correlation_matrix.mean(axis=0))


    for avg, layer in zip(avgs, layers_to_use):
        plt.plot(avg, label='Layer {}'.format(layer))

    plt.xlabel('Class Index (sorted by correlation)')
    plt.ylabel('Average Correlation')
    plt.ylim(0, 0.55)
    plt.title('Average Sorted Correlation for Each Layer')
    F.XX(0.5,1.5)
    plt.legend()
    F.save('average_sorted_correlation_{}'.format(datatype))

    for avg, layer in zip(avgs, layers_to_use):
        plt.plot(avg, label='Layer {}'.format(layer))

    plt.xlim(0,50)
    plt.xlabel('Class Index (sorted by correlation)')
    plt.ylabel('Average Correlation')
    plt.title('Average Sorted Correlation for Each Layer')
    F.XX(0.5,0.5)
    plt.legend()
    F.save('zoom_average_sorted_correlation_{}'.format(datatype))

