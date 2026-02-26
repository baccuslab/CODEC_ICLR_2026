import h5py as h5
import tqdm
from IPython import embed
import numpy as np
import bscope
import bscope.ic as bic
import glob
import os

path = '/data/codec/decomps/resnet50_act_normgrad_top_3_nosoftmax/'
final_model_name = 'sae_full_model_epoch_300.pt'
datapath = os.path.join(path, 'data.h5')
SYN_COMPUTED=False

sweep_name = 'sae-sigthresh-sweep'
saepath = os.path.join(path, 'saes', sweep_name + '/')

layer_dirs = glob.glob(os.path.join(saepath, '*'))
for layer_dir in layer_dirs:
    sae_dirs = glob.glob(os.path.join(layer_dir, '*'))
    for sae_dir in tqdm.tqdm(sae_dirs):
        try:
            sae_path = os.path.join(sae_dir, final_model_name)

            mode_summary_path = os.path.join(sae_dir, 'mode_summary.h5')
            file = h5.File(mode_summary_path, 'w')

            device= 'cuda:3'

            file.create_group('layers')
            file.create_group('layers/15')

            data, targets = bic.load_contribution_data(datapath, 'contributions', '15', 'sum')
            syn = bic.SemanticAnalyzer('/data/codec/hierarchy_metadata/semantic_indexes.json')
            mask_mtx, mask_labels = syn.get_all_semantic_masks(targets)
            file.create_dataset('mask_labels', data=mask_labels)
            file.create_dataset('mask_matrix', data=mask_mtx)

            sae, loadings, dictionary, data_agg, reconstructed_agg, r2 = bscope.load_sae(sae_path, data, device=device, eval_mode=True)


            corr_mtx = bscope.mtx_corr(loadings, mask_mtx.T)
            
            print('------------------')
            print(sae_path)
            print(r2)
            print(np.sum(corr_mtx>0.2))
            print(dictionary.shape)
            
            file['layers/15'].create_dataset('corr_mtx', data=corr_mtx)
            file['layers/15'].create_dataset('r2', data=r2)
            file['layers/15'].create_dataset('loadings', data=loadings)
            file['layers/15'].create_dataset('dictionary', data=dictionary)
            file['layers/15'].create_dataset('data_agg', data=data_agg)
            file['layers/15'].create_dataset('reconstructed_agg', data=reconstructed_agg)
        except:
            print("Could not process", sae_dir)

