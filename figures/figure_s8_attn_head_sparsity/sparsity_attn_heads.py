import bscope.ic as bic
import bscope
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from fycus import Fycus
import h5py
import scipy.stats as stats

cont_algo_str = "int_grad_top_1_True_steps_10"
model_type = 'vit'
layer_types = ['attention']
cont_spatial = 'sum'
act_spatial = 'sum'

N_HEADS = 12
N_TRIALS = 1
VERBOSE = False

F = Fycus('sparsity_attn_heads', extension='svg')

for layer_type in layer_types:
    FIG = f'figure_s_vit_sparsity_attn_heads_{cont_algo_str}_{model_type}_{layer_type}'

    path= f'/data/codec/h5s/int_grad_top_1_True_vit_{layer_type}_steps_10/data.h5'
    #path = f'/data/codec/h5s/act_normgrad_top_1_True_vit_{layer_type}/data.h5'



    # First get the number of layers
    with h5py.File(path, 'r') as f:
        layers = list(f.keys())
        num_layers = len(layers) - 1 # Exclude 'targets' key
        print(f"Found {num_layers} layers: {layers}")



    # 'mean' and 'sem' is over trials
    rnd_heads_act_hoyers_mean = []
    rnd_heads_con_hoyers_mean = []

    rnd_heads_act_hoyers_sem = []
    rnd_heads_con_hoyers_sem = []

    # Variation, defined here as sem/mean
    rnd_heads_act_hoyers_variation = []
    rnd_heads_con_hoyers_variation = []

    # Learned heads
    learned_heads_act_hoyers = []
    learned_heads_con_hoyers = []

    # Difference between learned and random heads
    heads_act_hoyers_difference_mean = []
    heads_con_hoyers_difference_mean = []


    for l in range(num_layers):
        data = bic.load_contribution_data(path, 'contributions', l, cont_spatial)[0]
        act_data = bic.load_contribution_data(path, 'activations', l, act_spatial)[0]
        
        # 'layer' means within this layer, save for individual trials
        rnd_heads_act_hoyers_layer = []
        rnd_heads_con_hoyers_layer = []
        heads_act_hoyers_difference_layer = []
        heads_con_hoyers_difference_layer = []
        
        for trial in range(N_TRIALS):
            # data has shape (samples, channels)
            perm = np.random.permutation(data.shape[1])
            rnd_data = data[:, perm]
            rnd_act_data = act_data[:, perm]

            # Sum within heads
            rnd_data = rnd_data.reshape(rnd_data.shape[0], N_HEADS, -1).sum(axis=-1)
            rnd_act_data = rnd_act_data.reshape(rnd_act_data.shape[0], N_HEADS, -1).sum(axis=-1)
            
            rnd_heads_A_hoyer = bscope.hoyer(rnd_act_data)
            rnd_heads_C_hoyer = bscope.hoyer(rnd_data) # Shape: (samples, heads)
            
            rnd_heads_act_hoyers_layer.append(rnd_heads_A_hoyer)
            rnd_heads_con_hoyers_layer.append(rnd_heads_C_hoyer)
        
        # Compute sparsity over learned heads
        data = data.reshape(data.shape[0], N_HEADS, -1).sum(axis=-1)
        act_data = act_data.reshape(act_data.shape[0], N_HEADS, -1).sum(axis=-1)

        learned_heads_A_hoyer = bscope.hoyer(act_data)
        learned_heads_C_hoyer = bscope.hoyer(data) # Shape: (samples, heads)

        learned_heads_act_hoyers.append(learned_heads_A_hoyer)
        learned_heads_con_hoyers.append(learned_heads_C_hoyer)

        # Compute various quantities for random heads
        rnd_heads_act_hoyers_layer = np.array(rnd_heads_act_hoyers_layer)
        rnd_heads_con_hoyers_layer = np.array(rnd_heads_con_hoyers_layer)
        # Now has shape (trials, samples, heads)

        heads_act_hoyers_difference_layer = learned_heads_A_hoyer - rnd_heads_act_hoyers_layer
        heads_con_hoyers_difference_layer = learned_heads_C_hoyer - rnd_heads_con_hoyers_layer
        
        rnd_heads_act_hoyers_mean.append(rnd_heads_act_hoyers_layer.mean(axis=0))
        rnd_heads_con_hoyers_mean.append(rnd_heads_con_hoyers_layer.mean(axis=0))
        rnd_heads_act_hoyers_sem.append(stats.sem(rnd_heads_act_hoyers_layer, axis=0))
        rnd_heads_con_hoyers_sem.append(stats.sem(rnd_heads_con_hoyers_layer, axis=0))
        heads_act_hoyers_difference_mean.append(heads_act_hoyers_difference_layer.mean(axis=0))
        heads_con_hoyers_difference_mean.append(heads_con_hoyers_difference_layer.mean(axis=0))
        # Now has shape (samples, heads)
        rnd_heads_act_hoyers_variation.append(rnd_heads_act_hoyers_sem[-1] / rnd_heads_act_hoyers_mean[-1])
        rnd_heads_con_hoyers_variation.append(rnd_heads_con_hoyers_sem[-1] / rnd_heads_con_hoyers_mean[-1])

        print(data.shape)


    if VERBOSE:
        print ("Making the following plots in addition: \
            Random Split Activation-Based Attention Head Sparsity Mean per Layer \
            Random Split Activation-Based Attention Head Sparsity SEM per Layer \
            Random Split Contribution-Based Attention Head Sparsity Mean per Layer \
            Random Split Contribution-Based Attention Head Sparsity SEM per Layer \
            Random Split Activation-Based Attention Head Sparsity Variation per Layer \
            Random Split Contribution-Based Attention Head Sparsity Variation per Layer\
            Learned vs. Random Split Activation-Based Attention Head Sparsity per Layer \
            Learned vs. Random Split Contribution-Based Attention Head Sparsity per Layer")
        
        # Mean, act
        plt.figure()
        plt.boxplot(rnd_heads_act_hoyers_mean, showfliers=False, boxprops=dict(color='b'), medianprops=dict(color='b'))
        # Dynamically set x-axis ticks based on number of layers
        if num_layers <= 1:
            ticks = [0]
        else:
            ticks = np.linspace(0, num_layers - 1, num=5)
            ticks = np.round(ticks).astype(int)
            ticks = np.unique(ticks)
        plt.xticks(ticks, [str(t) for t in ticks])
        plt.ylim(0.0, 1.0)
        plt.ylabel('Sparsity (Hoyer)')
        plt.xlabel('Layer')
        plt.title('Random Split Activation-Based Attention Head Sparsity Mean per Layer')
        plt.tight_layout()
        F.QT()
        F.save(f'{FIG}_rnd_heads_act_hoyers_mean_per_layer')
        plt.close()



        # SEM, act
        plt.figure()
        plt.boxplot(rnd_heads_act_hoyers_sem, showfliers=False, boxprops=dict(color='b'), medianprops=dict(color='b'))
        # Dynamically set x-axis ticks based on number of layers
        if num_layers <= 1:
            ticks = [0]
        else:
            ticks = np.linspace(0, num_layers - 1, num=5)
            ticks = np.round(ticks).astype(int)
            ticks = np.unique(ticks)
        plt.xticks(ticks, [str(t) for t in ticks])
        plt.ylim(0.0, 1.0)
        plt.ylabel('Sparsity (Hoyer)')
        plt.xlabel('Layer')
        plt.title('Random Split Activation-Based Attention Head Sparsity SEM per Layer')
        plt.tight_layout()
        F.QT()
        F.save(f'{FIG}_rnd_heads_act_hoyers_sem_per_layer')
        plt.close()




        # Mean, con
        plt.figure()
        plt.boxplot(rnd_heads_con_hoyers_mean, showfliers=False, boxprops=dict(color='k'), medianprops=dict(color='k'))
        # Dynamically set x-axis ticks based on number of layers
        if num_layers <= 1:
            ticks = [0]
        else:
            ticks = np.linspace(0, num_layers - 1, num=5)
            ticks = np.round(ticks).astype(int)
            ticks = np.unique(ticks)
        plt.xticks(ticks, [str(t) for t in ticks])
        plt.ylim(0.0, 1.0)
        plt.ylabel('Sparsity (Hoyer)')
        plt.xlabel('Layer')
        plt.title('Random Split Contribution-Based Attention Head Sparsity Mean per Layer')
        plt.tight_layout()
        F.QT()
        F.save(f'{FIG}_rnd_heads_con_hoyers_mean_per_layer')
        plt.close()




        # SEM, con
        plt.figure()
        plt.boxplot(rnd_heads_con_hoyers_sem, showfliers=False, boxprops=dict(color='k'), medianprops=dict(color='k'))
        # Dynamically set x-axis ticks based on number of layers
        if num_layers <= 1:
            ticks = [0]
        else:
            ticks = np.linspace(0, num_layers - 1, num=5)
            ticks = np.round(ticks).astype(int)
            ticks = np.unique(ticks)
        plt.xticks(ticks, [str(t) for t in ticks])
        plt.ylim(0.0, 1.0)
        plt.ylabel('Sparsity (Hoyer)')
        plt.xlabel('Layer')
        plt.title('Random Split Contribution-Based Attention Head Sparsity SEM per Layer')
        plt.tight_layout()
        F.QT()
        F.save(f'{FIG}_rnd_heads_con_hoyers_sem_per_layer')
        plt.close()



        # Variation, act
        plt.figure()
        plt.boxplot(rnd_heads_act_hoyers_variation, showfliers=False, boxprops=dict(color='b'), medianprops=dict(color='b'))
        # Dynamically set x-axis ticks based on number of layers
        if num_layers <= 1:
            ticks = [0]
        else:
            ticks = np.linspace(0, num_layers - 1, num=5)
            ticks = np.round(ticks).astype(int)
            ticks = np.unique(ticks)
        plt.xticks(ticks, [str(t) for t in ticks])
        plt.ylim(0.0, 1.0)
        plt.ylabel('Sparsity (Hoyer)')
        plt.xlabel('Layer')
        plt.title('Random Split Activation-Based Attention Head Sparsity Variation per Layer')
        plt.tight_layout()
        F.QT()
        F.save(f'{FIG}_rnd_heads_act_hoyers_variation_per_layer')
        plt.close()



        # Variation, con
        plt.figure()
        plt.boxplot(rnd_heads_con_hoyers_variation, showfliers=False, boxprops=dict(color='k'), medianprops=dict(color='k'))
        # Dynamically set x-axis ticks based on number of layers
        if num_layers <= 1:
            ticks = [0]
        else:
            ticks = np.linspace(0, num_layers - 1, num=5)
            ticks = np.round(ticks).astype(int)
            ticks = np.unique(ticks)
        plt.xticks(ticks, [str(t) for t in ticks])
        plt.ylim(0.0, 1.0)
        plt.ylabel('Sparsity (Hoyer)')
        plt.xlabel('Layer')
        plt.title('Random Split Contribution-Based Attention Head Sparsity Variation per Layer')
        plt.tight_layout()
        F.QT()
        F.save(f'{FIG}_rnd_heads_con_hoyers_variation_per_layer')
        plt.close()



        # Act, learned vs. random
        plt.figure()
        plt.boxplot(learned_heads_act_hoyers, positions=np.arange(num_layers)-0.1, showfliers=False, boxprops=dict(color='b'), medianprops=dict(color='b'))
        plt.boxplot(rnd_heads_act_hoyers_mean, positions=np.arange(num_layers)+0.1, showfliers=False, boxprops=dict(color='b', alpha=0.5), medianprops=dict(color='b', alpha=0.5))
        # Dynamically set x-axis ticks based on number of layers
        if num_layers <= 1:
            ticks = [0]
        else:
            ticks = np.linspace(0, num_layers - 1, num=5)
            ticks = np.round(ticks).astype(int)
            ticks = np.unique(ticks)
        plt.xticks(ticks, [str(t) for t in ticks])
        plt.ylim(0.0, 1.0)
        plt.ylabel('Sparsity (Hoyer)')
        plt.xlabel('Layer')
        plt.title('Learned vs. Random Split Activation-Based Attention Head Sparsity per Layer')
        plt.tight_layout()
        F.QT()
        F.save(f'{FIG}_learned_vs_rnd_heads_act_hoyers_per_layer')
        plt.close()



        # Con, learned vs. random
        plt.figure()
        plt.boxplot(learned_heads_con_hoyers, positions=np.arange(num_layers)-0.1, showfliers=False, boxprops=dict(color='k'), medianprops=dict(color='k'))
        plt.boxplot(rnd_heads_con_hoyers_mean, positions=np.arange(num_layers)+0.1, showfliers=False, boxprops=dict(color='k', alpha=0.5), medianprops=dict(color='k', alpha=0.5))
        # Dynamically set x-axis ticks based on number of layers
        if num_layers <= 1:
            ticks = [0]
        else:
            ticks = np.linspace(0, num_layers - 1, num=5)
            ticks = np.round(ticks).astype(int)
            ticks = np.unique(ticks)
        plt.xticks(ticks, [str(t) for t in ticks])
        plt.ylim(0.0, 1.0)
        plt.ylabel('Sparsity (Hoyer)')
        plt.xlabel('Layer')
        plt.title('Learned vs. Random Split Contribution-Based Attention Head Sparsity per Layer')
        plt.tight_layout()
        F.QT()
        F.save(f'{FIG}_learned_vs_rnd_heads_con_hoyers_per_layer')
        plt.close()




    # Con vs. act, learned heads
    plt.figure()
    plt.boxplot(learned_heads_con_hoyers, positions=np.arange(num_layers)-0.1, showfliers=False, boxprops=dict(color='k'), medianprops=dict(color='k'))
    plt.boxplot(learned_heads_act_hoyers, positions=np.arange(num_layers)+0.1, showfliers=False, boxprops=dict(color='b'), medianprops=dict(color='b'))
    # Dynamically set x-axis ticks based on number of layers
    if num_layers <= 1:
        ticks = [0]
    else:
        ticks = np.linspace(0, num_layers - 1, num=5)
        ticks = np.round(ticks).astype(int)
        ticks = np.unique(ticks)
    plt.xticks(ticks, [str(t) for t in ticks])
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.ylim(0.0, 1.0)
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    plt.ylabel('Sparsity (Hoyer)')
    plt.xlabel('Layer')
    plt.title('Learned Attention Head Sparsity: Contribution vs. Activation')
    plt.tight_layout()
    F.QT()
    F.save(f'{FIG}_learned_heads_con_vs_act_hoyers_per_layer')
    plt.close()



    # Con vs. act, random heads
    plt.figure()
    plt.boxplot(rnd_heads_con_hoyers_mean, positions=np.arange(num_layers)-0.1, showfliers=False, boxprops=dict(color='k'), medianprops=dict(color='k'))
    plt.boxplot(rnd_heads_act_hoyers_mean, positions=np.arange(num_layers)+0.1, showfliers=False, boxprops=dict(color='b'), medianprops=dict(color='b'))
    # Dynamically set x-axis ticks based on number of layers
    if num_layers <= 1:
        ticks = [0]
    else:
        ticks = np.linspace(0, num_layers - 1, num=5)
        ticks = np.round(ticks).astype(int)
        ticks = np.unique(ticks)
    plt.xticks(ticks, [str(t) for t in ticks])
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.ylim(0.0, 1.0)
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    plt.ylabel('Sparsity (Hoyer)')
    plt.xlabel('Layer')
    plt.title('Random Attention Head Sparsity: Contribution vs. Activation')
    plt.tight_layout()
    F.QT()
    F.save(f'{FIG}_rnd_heads_con_vs_act_hoyers_per_layer')
    plt.close()



    # Con vs. act, learned - random
    plt.figure()
    plt.boxplot(heads_con_hoyers_difference_mean, positions=np.arange(num_layers)-0.1, showfliers=False, boxprops=dict(color='k'), medianprops=dict(color='k'))
    plt.boxplot(heads_act_hoyers_difference_mean, positions=np.arange(num_layers)+0.1, showfliers=False, boxprops=dict(color='b'), medianprops=dict(color='b'))
    # Dynamically set x-axis ticks based on number of layers
    if num_layers <= 1:
        ticks = [0]
    else:
        ticks = np.linspace(0, num_layers - 1, num=5)
        ticks = np.round(ticks).astype(int)
        ticks = np.unique(ticks)
    plt.xticks(ticks, [str(t) for t in ticks])
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.ylim(-0.3, 0.7)
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    plt.ylabel('Sparsity (Hoyer)')
    plt.xlabel('Layer')
    plt.title('Learned - Random Split Attention Head Sparsity: Contribution vs. Activation')
    plt.tight_layout()
    F.QT()
    F.save(f'{FIG}_difference_con_vs_act_hoyers_per_layer')
    plt.close()