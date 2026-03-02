import bscope.ic as bic
import bscope
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from fycus import Fycus
import h5py

cont_algo_str = "int_grad_top_1_True_steps_10"
model_type = 'vit'
layer_types = ['block', 'attention', 'mlp']
cont_spatial = 'sum'
act_spatial = 'sum'

for layer_type in layer_types:
    #path = f'/data/codec/h5s/act_normgrad_top_1_True_vit_{layer_type}/data.h5'
    path = f'/data/codec/h5s/int_grad_top_1_True_{model_type}_{layer_type}_steps_10/data.h5'
    FIG = f'figure_s_vit_sparsity_{cont_algo_str}_{model_type}_{layer_type}'
    F = Fycus('sparsity_vit', extension='svg')

    act_hoyers = []
    con_hoyers = []

    act_hoyers_std = []
    con_hoyers_std = []

    bcon_hoyers = []
    bact_hoyers = []

    # First get the number of layers
    with h5py.File(path, 'r') as f:
        layers = list(f.keys())
        num_layers = len(layers) - 1 # Exclude 'targets' key
        print(f"Found {num_layers} layers: {layers}")

    for l in range(num_layers):
        data = bic.load_contribution_data(path, 'contributions', l, cont_spatial)[0]
        act_data = bic.load_contribution_data(path, 'activations', l, act_spatial)[0]

        A =bscope.hoyer(act_data)
        C = bscope.hoyer(data)
        
        bcon_hoyers.append(C)
        bact_hoyers.append(A)
        print(data.shape)
        print(A.shape)


            # Make the bins increments of 0.05 from 0 to 1
        bins = np.arange(0, 1.05, 0.025)

        plt.figure()
        plt.hist(A, bins=bins, label='activations', color='b', alpha=0.9)
        plt.hist(C, bins=bins, label='contributions', color='k', alpha=0.9)
        plt.xlabel('Sparsity (hoyer)')
        plt.ylabel('Count')
        plt.tight_layout()
        F.QT()
        F.save(f'{FIG}_layer_{l}_sparsity_hist')
        plt.close()

        act_hoyers.append(np.nanmean(A))
        con_hoyers.append(np.nanmean(C))

        act_hoyers_std.append(np.nanstd(A))
        con_hoyers_std.append(np.nanstd(C))


    plt.figure()
    plt.errorbar(np.arange(num_layers), con_hoyers, yerr=con_hoyers_std, fmt='k')
    plt.errorbar(np.arange(num_layers), act_hoyers, yerr=act_hoyers_std, fmt='b')
    plt.title('Average Sparsity (Hoyer) per Layer')

    plt.legend(['contributions', 'activations'])
    plt.ylabel('Sparsity (hoyer)')
    plt.xlabel('layer')
    plt.tight_layout()
    F.QT()
    F.save(f'{FIG}_avg_sparsity_per_layer')
    plt.close()

    plt.figure()
    # print(len(con_hoyers))
    # print(np.arange(num_layers).shape)
    # Make bcon black and bact blue
    plt.boxplot(bcon_hoyers, positions=np.arange(num_layers)-0.1, showfliers=False, boxprops=dict(color='k'), medianprops=dict(color='k'))
    plt.boxplot(bact_hoyers, positions=np.arange(num_layers)+0.1, showfliers=False, boxprops=dict(color='b'), medianprops=dict(color='b'))
    # Dynamically set x-axis ticks based on number of layers
    if num_layers <= 1:
        ticks = [0]
    else:
        ticks = np.linspace(0, num_layers - 1, num=5)
        ticks = np.round(ticks).astype(int)
        ticks = np.unique(ticks)
    plt.xticks(ticks, [str(t) for t in ticks])
    plt.ylim(0, 1.0)
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    plt.ylabel('Sparsity (Hoyer)')
    plt.xlabel('Layer')
    plt.tight_layout()
    F.QT()
    F.save(f'{FIG}_boxplot_sparsity_per_layer')
    plt.close()
