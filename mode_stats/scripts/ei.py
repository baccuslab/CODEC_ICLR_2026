import h5py as h5
import skkm
import os
from IPython import embed
import tqdm
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import bscope
import bscope.ic as bic
from fycus import Fycus

FIG = 'final_figure_3'
F = Fycus(FIG, extension='svg')

# path = '/data/h5s/act_normgrad_top_1_False_resnet50/data.h5'
# path = '/data/h5s/act_normgrad_top_1_True_resnet50/data.h5'
path = '/data/h5s/int_grad_top_1_True_resnet50_steps_10/data.h5'
datapath = path

def compute_rowwise_correlations(N, P):
    """
    Compute Pearson correlation between corresponding rows of two matrices.
    
    Args:
        N: numpy array of shape (50000, 256)
        P: numpy array of shape (50000, 256)
    
    Returns:
        correlations: numpy array of shape (50000,) containing correlation coefficients
    """
    # Method 1: Vectorized computation (fastest)
    def vectorized_correlation(N, P):
        # Center the data (subtract mean along axis 1)
        N_centered = N - N.mean(axis=1, keepdims=True)
        P_centered = P - P.mean(axis=1, keepdims=True)
        
        # Compute numerator (covariance)
        numerator = np.sum(N_centered * P_centered, axis=1)
        
        # Compute denominator (product of standard deviations)
        N_std = np.sqrt(np.sum(N_centered**2, axis=1))
        P_std = np.sqrt(np.sum(P_centered**2, axis=1))
        denominator = N_std * P_std
        
        # Avoid division by zero
        correlations = np.divide(numerator, denominator, 
                               out=np.zeros_like(numerator), 
                               where=denominator!=0)
        
        return correlations
    
    return vectorized_correlation(N, P)


img = 75*50+4

corrs = []
for i in range(16):
    print(f'Processing layer {i}')
    P = bic.load_contribution_data(datapath, 'contributions', i, 'positive')[0]
    N = bic.load_contribution_data(datapath, 'contributions', i, 'negative')[0]
    SUM = bic.load_contribution_data(datapath, 'contributions', i, 'sum')[0]
    correlations = compute_rowwise_correlations(N, P)
    
    corrs.append(correlations*-1)
    
    max_P = np.max(np.abs(P[img]))
    max_N = np.max(np.abs(N[img]))
    lim = np.max([max_P, max_N])

    OFFSET = lim*0.05

    # Change markersize to 1
    plt.scatter(P[img], N[img], alpha=0.3, s=5)
    plt.xlim(0-OFFSET, lim)
    plt.ylim(-lim, 0+OFFSET)
    plt.gca().set_aspect('equal', 'box')
    plt.ylabel('Negative Contributions')
    plt.xlabel('Positive Contributions')
    plt.xticks(np.arange(0, lim+1, 5))
    plt.title(f'Layer {i} Contributions')
    F.QT()
    F.save(f'layer_{i}_contribution_scatter')
    

    contribution_range = np.max(SUM,0)-np.min(SUM,0)
    plt.hist(contribution_range, bins=50)
    F.QT()
    F.save(f'layer_{i}_contribution_range_hist')

    chan_mins = np.min(SUM,0)
    for chan in np.argsort(chan_mins)[:3]:
        fig, ax = plt.subplots(1,2)
        # make y axis log scale
        
        plt.sca(ax[0])
        plt.hist(SUM[:, chan], bins=50)
        plt.gca().set_yscale('log')
        plt.axvline(0, color='k', linestyle='--')
        plt.title(f'Layer {i} Channel {chan} Contribution')

        plt.sca(ax[1])
        plt.plot(SUM[:, chan])
        F.save(f'layer_{i}_channel_{chan}_contribution_misc')




    for chan in np.argsort(contribution_range)[-2:]:
        plt.plot(SUM[:, chan])
        plt.title(f'Layer {i} Channel {chan} Contribution Range')
        plt.xlabel('Image Index')
        plt.ylabel('Contribution')
        F.QT()
        F.save(f'layer_{i}_channel_{chan}_contribution_range')


    IDXS = []
    for j in [232, 285]:
        IDXS.extend(np.arange(j*50, (j+1)*50))

    plt.imshow(SUM[IDXS].T, aspect='auto', cmap='bwr', vmin=-np.max(np.abs(SUM))*0.05, vmax=np.max(np.abs(SUM))*0.05, interpolation='none')
    plt.colorbar(label='Contribution')
    F.QT()
    F.save(f'layer_{i}_contribution_heatmap')


plt.boxplot(corrs, positions=np.arange(len(corrs)), showfliers=False)
plt.xticks(np.arange(len(corrs)), np.arange(len(corrs)))
plt.xlabel('Layer')
plt.ylabel('Correlation (r)')
plt.xticks(np.arange(0, 16, 4))
F.QT()
F.save('contribution_correlation_boxplot')



