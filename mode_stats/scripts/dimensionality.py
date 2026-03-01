"""

Example figure script demonstrating the standardized setup.
Place scripts like this in subdirectories: figure_1/, figure_2/, etc.
"""
import tqdm
import scipy
import bscope.ic as bic
from sklearn.decomposition import PCA
import bscope
import numpy as np
import matplotlib.pyplot as plt
import sys
from IPython import embed
from fycus import Fycus

FIG = 'final_figure_3_net_act'
PATH = '/data/h5s/int_grad_top_1_False_resnet50_steps_10/data.h5'


CSPATIAL_OP = 'sum'
ASPATIAL_OP = 'positive'
thresh = 0.95

F = Fycus(FIG)

num_to_thresh = []
num_channels = []

for LAYER in np.arange(0,16):
    cdata = bic.load_contribution_data(PATH, 'contributions', LAYER, CSPATIAL_OP)[0]
    adata = bic.load_contribution_data(PATH, 'activations', LAYER, ASPATIAL_OP)[0]

    masks, labels = bic.get_masks(leaf_only=True)

    print(masks.shape)
    
    cmn = []
    amn = []

    for i, mask in enumerate(masks):
        C = cdata[mask]
        A = adata[mask]

        cmn.append(np.mean(C, axis=0))
        amn.append(np.mean(A, axis=0))

        
    cmn = np.array(cmn) # 1000 by channels
    amn = np.array(amn)
    scale=0.2

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(cmn, aspect='auto', cmap='PiYG', interpolation='none', clim=(-np.max(cmn)*scale,np.max(cmn)*scale))
    ax[0].set_title('Mean contributions per class')
    ax[0].set_ylabel('Classes')
    ax[0].set_xlabel('Channels')
    ax[1].imshow(amn, aspect='auto', cmap='PiYG', interpolation='none', clim=(-np.max(amn)*scale,np.max(amn)*scale))
    ax[1].set_title('Mean activations per class')
    ax[1].set_ylabel('Classes')
    ax[1].set_xlabel('Channels')
    F.QH()
    F.save(f'layer_{LAYER}_mean_contributions_activations')
    
    crs = bscope.mtx_corr(cmn.T, cmn.T)
    plt.imshow(crs, cmap='PiYG', clim=(-1,1), interpolation='none')
    plt.colorbar()
    F.QT()
    F.save(f'layer_{LAYER}_contribution_corr')

    scrs = []
    for i in range(crs.shape[0]):
        s = np.sort(crs[i,:])[::-1]
        scrs.append(s)

    scrs = np.array(scrs)

    ars = bscope.mtx_corr(amn.T, amn.T)
    plt.imshow(ars, cmap='PiYG', clim=(-1,1), interpolation='none')
    plt.colorbar()
    F.QT()
    F.save(f'layer_{LAYER}_activations_corr')

    ascrs = []
    for i in range(ars.shape[0]):
        s = np.sort(ars[i,:])[::-1]
        ascrs.append(s)
    
    plt.imshow(scrs, cmap='PiYG', clim=(-1,1), interpolation='none')
    plt.colorbar()
    F.QT()
    F.save(f'layer_{LAYER}_sorted_contribution_corr')
    plt.imshow(ascrs, cmap='PiYG', clim=(-1,1), interpolation='none')
    plt.colorbar()
    F.QT()
    F.save(f'layer_{LAYER}_sorted_activations_corr')
    plt.plot(np.mean(scrs, axis=0), label='Contributions')
    plt.plot(np.mean(ascrs, axis=0), label='Activations')
    plt.legend()
    F.QT()
    F.save(f'layer_{LAYER}_mean_sorted_corr')

    plt.plot(crs[0])
    plt.plot(ars[0])
    F.QT()
    F.save(f'layer_{LAYER}_example_corr')
    
    plt.plot(cmn[0], label='Contributions')
    plt.plot(cmn[899], label='Contributions')
    F.QT()
    F.save(f'layer_{LAYER}_example_contributions')

    plt.plot(amn[0], label='Activations')
    plt.plot(amn[899], label='Activations')
    F.QT()
    F.save(f'layer_{LAYER}_example_activations')
    # # Find two classes that are very different in contributions but similar in activations 


    # w = np.where((crs > 0.4) & (ars< 0.1))
    # class_1 = w[0][0]
    # class_2 = w[1][0]

    # plt.plot(cmn[class_1], label=f'Class {class_1}')
    # plt.plot(cmn[class_2], label=f'Class {class_2}')
    # plt.show()

    # plt.plot(amn[class_1], label=f'Class {class_1}')
    # plt.plot(amn[class_2], label=f'Class {class_2}')


    print(cmn.shape, amn.shape)
    pca_c = PCA()
    pca_a = PCA()

    # cmn = scipy.stats.zscore(cmn, axis=0)
    # amn = scipy.stats.zscore(amn, axis=0)

    cmn = np.nan_to_num(cmn)
    amn = np.nan_to_num(amn)
    
    # zcmn = cmn - np.mean(cmn, axis=0)
    # zamn = amn - np.mean(amn, axis=0)

    pca_c.fit_transform(cmn)
    pca_a.fit_transform(amn)

    expvar_c = pca_c.explained_variance_ratio_
    expvar_a = pca_a.explained_variance_ratio_
    
    cumvar_c = np.cumsum(expvar_c)
    cumvar_a = np.cumsum(expvar_a)
    # Find how many components are needed to explain 90% of the variance

    # How many components to reach the threshold?
    num_c = np.argmax(cumvar_c >= thresh) + 1
    num_a = np.argmax(cumvar_a >= thresh) + 1

    num_chan = adata.shape[1]
    num_channels.append(num_chan)
    num_to_thresh.append([num_c, num_a])

    plt.plot(cumvar_c, label='Contributions')
    plt.plot(cumvar_a, label='Activations')
    plt.ylim([0,1])
    plt.legend()
    F.QT()
    F.save(f'layer_{LAYER}_pca_cumvar')


num_to_thresh = np.array(num_to_thresh)
c_num_to_thresh = num_to_thresh[:,0]
a_num_to_thresh = num_to_thresh[:,1]
num_channels = np.array(num_channels)

plt.plot(c_num_to_thresh / num_channels, label='Contributions')
plt.plot(a_num_to_thresh / num_channels, label='Activations')
F.QT()
F.save(f'pca_num_to_thresh_norm')

plt.plot(c_num_to_thresh, label='Contributions')
plt.plot(a_num_to_thresh, label='Activations')
F.QT()
F.save(f'pca_num_to_thresh')
