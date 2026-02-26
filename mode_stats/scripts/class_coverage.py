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

    class_avgs = []
    mode_avgs = []
    # layers_to_use=[0,4,9,14]
    layers_to_use=[2,6,12,15]
    for layer_idx, layer in enumerate(mode_summary.layers):
        if layer_idx not in layers_to_use:
            continue
        correlation_matrix = layer.imgnet_corr_mtx 
        num_modes = correlation_matrix.shape[0]
        num_classes = correlation_matrix.shape[1]

        assert num_classes == 1000, "Expected 1000 classes for ImageNet, got {}".format(num_classes)
        
        class_coverage = []
        for class_idx in range(num_classes):
            sorted_indices = np.argsort(correlation_matrix[:, class_idx])[::-1]
            sorted_class_corrs = correlation_matrix[sorted_indices, class_idx]
            class_coverage.append(sorted_class_corrs)

        class_coverage = np.array(class_coverage)  # Shape: (num_classes, num_modes)
        class_avgs.append(np.nanmean(class_coverage,0))  # Average correlation across classes for each mode

        mode_coverage = []
        for mode_idx in range(num_modes):
            sorted_indices = np.argsort(correlation_matrix[mode_idx, :])[::-1]
            sorted_mode_corrs = correlation_matrix[mode_idx, sorted_indices]
            mode_coverage.append(sorted_mode_corrs)
        mode_coverage = np.array(mode_coverage)  # Shape: (num_modes, num_classes)
        mode_avgs.append(np.nanmean(mode_coverage,0))  # Average correlation across classes for each mode

        


    # Plotting the average sorted CLASS correlation for each layer

    for avg, layer in zip(class_avgs, layers_to_use):
        plt.plot(avg, label='Layer {}'.format(layer))

    plt.xlabel('Class Index (sorted by correlation)')
    plt.ylabel('Average Correlation')
    plt.ylim(0, 0.55)

    plt.title('Average Sorted Correlation for Each Layer')
    F.XX(0.5,1.5)
    plt.legend()
    F.save('class_coverage_{}'.format(datatype))

    for avg, layer in zip(mode_avgs, layers_to_use):
        plt.plot(avg, label='Layer {}'.format(layer))
    plt.xlabel('Class Index (sorted by correlation)')
    plt.ylabel('Average Correlation')
    plt.ylim(0, 0.55)
    plt.title('Average Sorted Correlation for Each Layer')
    F.XX(0.5,1.5)
    plt.legend()
    F.save('mode_coverage_{}'.format(datatype))

    # Make same plots but zoomed into top 40
    for avg, layer in zip(class_avgs, layers_to_use):
        plt.plot(avg[:60], label='Layer {}'.format(layer))
    plt.xlabel('Mode Index (sorted by correlation)')
    plt.ylabel('Average Correlation')
    plt.ylim(0, 0.55)
    plt.title('Average Sorted Correlation for Each Layer (Top 60)')
    F.XX(0.5,1.0)
    plt.legend()
    F.save('class_coverage_top60_{}'.format(datatype))

    for avg, layer in zip(mode_avgs, layers_to_use):
        plt.plot(avg[:60], label='Layer {}'.format(layer))
    plt.xlabel('Mode Index (sorted by correlation)')
    plt.ylabel('Average Correlation')

    plt.ylim(0, 0.55)
    plt.title('Average Sorted Correlation for Each Layer (Top 60)')
    F.XX(0.5,1.0)
    plt.legend()
    F.save('mode_coverage_top60_{}'.format(datatype))


    


















