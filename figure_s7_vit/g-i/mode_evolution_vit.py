
# Apply standardized styling
import numpy as np
from scipy.stats import pearsonr
import bscope
import scipy.stats
import scipy.stats as stats
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from fycus import Fycus
from IPython import embed
import bscope.ic as bic
import os

cont_algo_str = "int_grad_top_1_True_steps_10"
model_type = 'vit'
layer_types = ['block', 'attention', 'mlp']
cont_spatial_operation = 'positive'  # 'sum', 'positive', 'negative'
act_spatial_operation = 'sum'  # 'sum', 'positive', 'negative'

F = Fycus('mode_evolution', extension='svg')

for layer_type in layer_types:
    
    FIG = f'figure_5_{cont_algo_str}_{model_type}_{layer_type}_cont_{cont_spatial_operation}_act_{act_spatial_operation}'

    c_ms_path = f'/data/codec/h5s/int_grad_top_1_True_{model_type}_{layer_type}_steps_10/saes/aleph_contributions_{cont_spatial_operation}/mode_summary.h5'
    c_ms = bic.ModeSummary(c_ms_path)

    a_ms_path = f'/data/codec/h5s/int_grad_top_1_True_{model_type}_{layer_type}_steps_10/saes/aleph_activations_{act_spatial_operation}/mode_summary.h5'
    a_ms = bic.ModeSummary(a_ms_path)

    basepath = os.path.split(os.path.split(os.path.split(c_ms_path)[0])[0])[0]
    datapath = os.path.join(basepath, 'data.h5')

    def avg_sorted_corr(corr_mtx):
        # N mode by N class
        corr_mtx= corr_mtx.T
        sorted_corrs = np.sort(corr_mtx, axis=1)[:, ::-1]
        return sorted_corrs 

    def avg_sorted_class_corrs(corr_mtx):
        # N mode by N class
        sorted_corrs = np.sort(corr_mtx, axis=1)[:, ::-1]
        return sorted_corrs 

    masks, labels = bic.get_masks(path='/data/codec/hierarchy_metadata/pruned_hierarchy.json')
    chunk_idxs = bic.chunk_masks(masks)
    imgnet_mask_matrix = c_ms.imgnet_mask_matrix

    c_atom_agg = []
    a_atom_agg = []
    c_loading_agg = []
    a_loading_agg = []
    c_mode_corr_agg = []
    a_mode_corr_agg = []
    c_chan_corr_agg = []
    a_chan_corr_agg = []
    c_num_modes_agg = []
    a_num_modes_agg = []
    c_num_corr_modes = []
    a_num_corr_modes = []

    # Layerwise comparison
    for L, (a_layer, c_layer) in enumerate(zip(a_ms.layers, c_ms.layers)):
        c_data = bic.load_contribution_data(datapath, 'contributions', L, 'positive')[0]
        a_data = bic.load_contribution_data(datapath, 'activations', L, 'positive')[0]

        c_num_modes_agg.append(c_layer.dictionary.shape[0])
        a_num_modes_agg.append(a_layer.dictionary.shape[0])

        c_chan_corr_mtx = bscope.mtx_corr(c_data, imgnet_mask_matrix.T)
        a_chan_corr_mtx = bscope.mtx_corr(a_data, imgnet_mask_matrix.T)
        c_mode_corr_mtx = c_layer.imgnet_corr_mtx  # mode by class
        a_mode_corr_mtx = a_layer.imgnet_corr_mtx  # mode by class

        c_max_mode_corrs = c_mode_corr_mtx.max(1)
        a_max_mode_corrs = a_mode_corr_mtx.max(1)
        
        c_num_corr_modes.append(np.sum(c_max_mode_corrs>0.2))
        a_num_corr_modes.append(np.sum(a_max_mode_corrs>0.2))
        print(c_max_mode_corrs.shape, a_max_mode_corrs.shape)

        c_max_chan_corrs = c_chan_corr_mtx.max(1)
        a_max_chan_corrs = a_chan_corr_mtx.max(1)

        c_mode_corr_agg.append((np.nanmedian(c_max_mode_corrs), scipy.stats.sem(c_max_mode_corrs, nan_policy='omit')))
        a_mode_corr_agg.append((np.nanmedian(a_max_mode_corrs), stats.sem(a_max_mode_corrs, nan_policy='omit')))
        c_chan_corr_agg.append((np.nanmedian(c_max_chan_corrs), scipy.stats.sem(c_max_chan_corrs, nan_policy='omit')))
        a_chan_corr_agg.append((np.nanmedian(a_max_chan_corrs), scipy.stats.sem(a_max_chan_corrs, nan_policy='omit')))

        plt.figure()
        plt.hist(c_max_chan_corrs, color='k', bins=50, alpha=0.5, label='contributions', density=True)
        plt.hist(a_max_chan_corrs, color='b', bins=50, alpha=0.5, label='activations', density=True)
        plt.axvline(np.nanmedian(c_max_chan_corrs), color='k', ls='--')
        plt.axvline(np.nanmedian(a_max_chan_corrs), color='b', ls='--')
        plt.xlabel('Channel top corr')
        plt.ylabel('Density')
        plt.xlim([0,1.0])
        plt.tight_layout()
        F.QT()
        F.save(f'{FIG}_chan_corrs_hist_layer_{L}')
        #plt.savefig(f'{FIG}_chan_corrs_hist_layer_{L}.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.hist(c_max_mode_corrs, color='k', bins=50, alpha=0.5, label='contributions', density=True)
        plt.hist(a_max_mode_corrs, color='b', bins=50, alpha=0.5, label='activations', density=True)
        plt.axvline(np.nanmedian(c_max_mode_corrs), color='k', ls='--')
        plt.axvline(np.nanmedian(a_max_mode_corrs), color='b', ls='--')
        plt.xlabel('Mode top corr')
        plt.ylabel('Density')
        plt.xlim([0,1.0])
        plt.tight_layout()
        F.QT()
        F.save(f'{FIG}_mode_corrs_hist_layer_{L}')
        #plt.savefig(f'{FIG}_mode_corrs_hist_layer_{L}.png', dpi=300, bbox_inches='tight')
        plt.close()

        a_loadings = a_layer.loadings
        c_loadings = c_layer.loadings

        a_atoms = a_layer.dictionary
        c_atoms = c_layer.dictionary

        print(a_loadings.shape, c_loadings.shape)
        print(a_atoms.shape, c_atoms.shape)

        c_atoms_hoyer = bscope.hoyer(c_atoms) 
        a_atoms_hoyer = bscope.hoyer(a_atoms)
        c_loadings_hoyer = bscope.hoyer(c_loadings.T)
        a_loadings_hoyer = bscope.hoyer(a_loadings.T)


        c_atoms_hoyer_mean = np.nanmedian(c_atoms_hoyer)
        c_atoms_hoyer_sem = scipy.stats.sem(c_atoms_hoyer, nan_policy='omit')

        c_loadings_hoyer_mean = np.nanmedian(c_loadings_hoyer)
        c_loadings_hoyer_sem = scipy.stats.sem(c_loadings_hoyer, nan_policy='omit')

        a_atoms_hoyer_sem = scipy.stats.sem(a_atoms_hoyer, nan_policy='omit')
        a_atoms_hoyer_mean = np.nanmedian(a_atoms_hoyer)

        a_loadings_hoyer_mean = np.nanmedian(a_loadings_hoyer)
        a_loadings_hoyer_sem = scipy.stats.sem(a_loadings_hoyer, nan_policy='omit')

        a_atom_agg.append((a_atoms_hoyer_mean, a_atoms_hoyer_sem))
        c_atom_agg.append((c_atoms_hoyer_mean, c_atoms_hoyer_sem))
        a_loading_agg.append((a_loadings_hoyer_mean, a_loadings_hoyer_sem))
        c_loading_agg.append((c_loadings_hoyer_mean, c_loadings_hoyer_sem))


    # Plot atom hoyers with error bars
    c_atom_hoyers = c_atom_agg
    a_atom_hoyers = a_atom_agg
    c_loading_hoyers = c_loading_agg
    a_loading_hoyers = a_loading_agg


    atom_x = np.arange(len(c_atom_hoyers))
    c_atom_y = [h[0] for h in c_atom_hoyers]
    c_atom_err = [h[1] for h in c_atom_hoyers]
    a_atom_y = [h[0] for h in a_atom_hoyers]
    a_atom_err = [h[1] for h in a_atom_hoyers]
    plt.figure()
    plt.errorbar(atom_x, c_atom_y, yerr=c_atom_err, label='contributions', fmt='o-', capsize=5)
    plt.errorbar(atom_x, a_atom_y, yerr=a_atom_err, label='activations', fmt='o-', capsize=5)
    plt.xticks(np.arange(0, len(c_atom_hoyers), 4))
    plt.xlabel('Layer')
    plt.ylabel('Hoyer')
    plt.title('Atom Hoyer by Layer')
    plt.legend()
    plt.tight_layout()
    F.QT()
    F.save(f'{FIG}_atom_hoyers_by_layer')
    #plt.savefig(f'{FIG}_atom_hoyers_by_layer.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot loading hoyers with error bars
    loading_x = np.arange(0, len(c_loading_hoyers))
    c_loading_y = [h[0] for h in c_loading_hoyers]
    c_loading_err = [h[1] for h in c_loading_hoyers]
    a_loading_y = [h[0] for h in a_loading_hoyers]
    a_loading_err = [h[1] for h in a_loading_hoyers]
    plt.figure()
    plt.errorbar(loading_x, c_loading_y, yerr=c_loading_err, label='contributions', fmt='o-', capsize=5)
    plt.errorbar(loading_x, a_loading_y, yerr=a_loading_err, label='activations', fmt='o-', capsize=5)
    plt.xticks(np.arange(0, len(c_atom_hoyers), 4))
    plt.xlabel('Layer')
    plt.ylabel('Hoyer')
    plt.title('Loading Hoyer by Layer')
    plt.legend()
    plt.tight_layout()
    F.QT()
    F.save(f'{FIG}_loading_hoyers_by_layer')
    #plt.savefig(f'{FIG}_loading_hoyers_by_layer.png', dpi=300, bbox_inches='tight')
    plt.close()

    mode_x = np.arange(len(c_mode_corr_agg))
    c_mode_y = [h[0] for h in c_mode_corr_agg]
    c_mode_err = [h[1] for h in c_mode_corr_agg]
    a_mode_y = [h[0] for h in a_mode_corr_agg]
    a_mode_err = [h[1] for h in a_mode_corr_agg]
    plt.figure()
    plt.errorbar(mode_x-0.1, c_mode_y, yerr=c_mode_err, label='contributions', fmt='o-', capsize=5)
    plt.errorbar(mode_x+0.1, a_mode_y, yerr=a_mode_err, label='activations', fmt='o-', capsize=5)
    plt.xticks(mode_x, mode_x)
    plt.xlabel('Layer')
    plt.ylabel('Top Mode Corr')
    plt.title('Top Mode Correlation by Layer')
    plt.legend()
    plt.ylim(0, 0.6)
    plt.tight_layout()
    F.QT()
    F.save(f'{FIG}_top_mode_corrs_by_layer')
    #plt.savefig(f'{FIG}_top_mode_corrs_by_layer.png', dpi=300, bbox_inches='tight')
    plt.close()

    chan_x = np.arange(len(c_chan_corr_agg))
    c_chan_y = [h[0] for h in c_chan_corr_agg]
    c_chan_err = [h[1] for h in c_chan_corr_agg]
    a_chan_y = [h[0] for h in a_chan_corr_agg]
    a_chan_err = [h[1] for h in a_chan_corr_agg]
    plt.figure()
    plt.errorbar(chan_x-0.1, c_chan_y, yerr=c_chan_err, label='contributions', fmt='o-', capsize=5)
    plt.errorbar(chan_x+0.1, a_chan_y, yerr=a_chan_err, label='activations', fmt='o-', capsize=5)
    plt.xticks(chan_x, chan_x)
    plt.xlabel('Layer')
    plt.ylabel('Top Channel Corr')
    plt.ylim(0, 1.0)
    #plt.ylim(0, 0.6)
    plt.title('Top Channel Correlation by Layer')
    plt.legend()
    plt.tight_layout()
    F.QT()
    F.save(f'{FIG}_top_chan_corrs_by_layer')
    #plt.savefig(f'{FIG}_top_chan_corrs_by_layer.png', dpi=300, bbox_inches='tight')
    plt.close()


    plt.figure()
    plt.plot(c_num_modes_agg, label='contributions')
    plt.plot(a_num_modes_agg, label='activations')
    plt.xlabel('Layer')
    plt.ylabel('Number of Modes')
    plt.title('Number of Modes by Layer')
    plt.legend()
    plt.tight_layout()
    F.QT()
    F.save(f'{FIG}_num_modes_by_layer')
    #plt.savefig(f'{FIG}_num_modes_by_layer.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot c_chan_y with fill_between error bars
    plt.figure()
    x= np.arange(len(c_num_modes_agg))
    plt.plot(x, c_chan_y, 'k', alpha=0.5, label='Contribution (chans)')
    # plt.fill_between(x, c_chan_y - np.array(c_chan_err), c_chan_y + np.array(c_chan_err), color='gray', alpha=0.25)

    plt.plot(x, a_chan_y, 'b', alpha=0.5, label='Activation (chans)')
    # plt.fill_between(x, a_chan_y - np.array(a_chan_err), a_chan_y + np.array(a_chan_err), color='blue', alpha=0.5)

    plt.plot(x, c_mode_y, 'k', label='Contribution (modes)')
    # plt.fill_between(x, c_mode_y - np.array(c_mode_err), c_mode_y + np.array(c_mode_err), color='gray', alpha=0.25)

    plt.plot(x, a_mode_y, 'b', label='Activation (modes)')
    # plt.fill_between(x, a_mode_y - np.array(a_mode_err), a_mode_y + np.array(a_mode_err), color='blue', alpha=0.5)
    plt.ylim(0, 1.0)
    #plt.ylim(0, 0.62)
    plt.yticks([0, .2, .4, .6, .8, 1.0])
    plt.xlabel('Layer')
    plt.ylabel('Avg. Corr.')
    plt.legend()
    plt.tight_layout()
    F.QT()
    F.save(f'{FIG}_top_corrs_by_layer_with_fill')
    #plt.savefig(f'{FIG}_top_corrs_by_layer_with_fill.svg', dpi=300, bbox_inches='tight')
    plt.close()





    x = np.arange(len(c_num_corr_modes))
    plt.figure()
    plt.plot(x, c_num_corr_modes, label='contributions')
    plt.plot(x, a_num_corr_modes, label='activations')
    plt.xlabel('Layer')
    plt.ylabel('# of correlated \n modes')
    plt.tight_layout()
    F.QT()
    F.save(f'{FIG}_num_corr_modes_by_layer')
    #plt.savefig(f'{FIG}_num_corr_modes_by_layer.png', dpi=300, bbox_inches='tight')
    plt.close()


