import h5py as h5
import numpy as np
import bscope
import bscope.ic as bic
import glob
import os
import torch

SAES_BASE_PATH = '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes'
device = 'cuda:0'

sweep_directories = glob.glob(os.path.join(SAES_BASE_PATH, 'sweep_int_grad_top_1_False_resnet50_steps_10_*_positive_STSAE'))
sweep_directories_mlp = glob.glob(os.path.join(SAES_BASE_PATH, 'sweep_int_grad_top_1_False_resnet50_steps_10_*_mlpsize_nonneg_STSAE'))

# Combine both sweep types
sweep_directories = sweep_directories + sweep_directories_mlp


datapath = '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/data.h5'

# Count total configs across all sweeps
total_configs = sum(len(glob.glob(os.path.join(sweep_dir, 'hypersweep_*'))) 
                    for sweep_dir in sweep_directories)
print(f'Found {len(sweep_directories)} sweep directories with {total_configs} total configs')

# Print breakdown per sweep
for sweep_dir in sweep_directories:
    config_dirs = glob.glob(os.path.join(sweep_dir, 'hypersweep_*'))
    print(f'  {os.path.basename(sweep_dir)}: {len(config_dirs)} hypersweeps')



datatype = 'contributions'
spatial_operation = 'positive'

for sweep_dir in sweep_directories:
    print('Searching sweep: {}'.format(os.path.basename(sweep_dir)))
    

    if 'aleph' in sweep_dir:
        config_directories = [sweep_dir]
    else:
        config_directories = glob.glob(os.path.join(sweep_dir, 'hypersweep_*'))
    

    
    print(f'Found {len(config_directories)} config directories')
    [print('>', os.path.basename(d)) for d in config_directories]
    
    for config_dir in config_directories:

        if os.path.exists(config_dir):
            print('+++Found config path:', config_dir)
        else:
            print('+++Didnt find config path:', config_dir)
            continue
        
        mode_summary_path = os.path.join(config_dir, 'mode_summary.h5')
        if os.path.exists(mode_summary_path):
            print(f'  Deleting: {os.path.basename(config_dir)}/mode_summary.h5')
            os.remove(mode_summary_path)
            print('+++Recomputing mode summary...')
        else:
            print('+++Creating new mode summary:', mode_summary_path)
        
        file = h5.File(mode_summary_path, 'w')

        
        SYN_COMPUTED = False
        file.create_group('layers')
        

        if 'aleph' in config_dir:
            sae_files = glob.glob(os.path.join(config_dir, 'layer_*.pt'))
        else:
            sae_files = glob.glob(os.path.join(config_dir, '*_final_model.pt'))

        for single_sae_path in sae_files:

            if 'aleph' in config_dir:
                layer = single_sae_path.split('layer_')[1].split('.pt')[0]
            else:
                layer = single_sae_path.split('_layer_')[1].split('_')[0]
            

            
            print('+++Using single_sae_path:', single_sae_path)
            print('+++Layer:', layer)
            file.create_group('layers/' + layer)
            
            data, targets = bic.load_contribution_data(datapath, datatype, layer, spatial_operation)
            
            if not SYN_COMPUTED:
                # Use the new get_masks function
                mask_matrix, mask_labels = bic.get_masks(
                    path='/mnt/data/codec/hierarchy_metadata/pruned_hierarchy.json',
                )
                file.create_dataset('mask_labels', data=mask_labels)
                file.create_dataset('mask_matrix', data=mask_matrix)
                print(mask_matrix.shape)
                
                imgnet_mask_matrix, imgnet_mask_labels = bic.get_masks(leaf_only=True, path='/mnt/data/codec/hierarchy_metadata/pruned_hierarchy.json',)
                file.create_dataset('imgnet_mask_labels', data=imgnet_mask_labels)
                file.create_dataset('imgnet_mask_matrix', data=imgnet_mask_matrix)
                print(imgnet_mask_matrix.shape)


                
                SYN_COMPUTED = True
            
            sae, loadings, dictionary, data_agg, reconstructed_agg, r2 = bscope.load_sae(single_sae_path, data, device=device, eval_mode=True)
            
            corr_mtx = bscope.mtx_corr(mask_matrix.T, loadings)
            imgnet_corr_mtx = bscope.mtx_corr(imgnet_mask_matrix.T, loadings)
            
            high_corrs = corr_mtx > 0.2
            high_features = np.sum(high_corrs, axis=0) > 0
            high_concepts = np.sum(high_corrs, axis=1) > 0
            high_features = np.sum(high_features)
            high_concepts = np.sum(high_concepts)
            
            print('------------------')
            print('+++Saving :', single_sae_path)
            print('+++R2:', r2)
            print('+++Number of correlations > 0.2:', np.sum(corr_mtx > 0.2))
            print('+++Number of features > 0.2:', high_features)
            print('+++Number of concepts > 0.2:', high_concepts)
            print('+++FeatureConceptRatio:', high_features / high_concepts)
            print('+++Dictionary shape:', dictionary.shape)
            
            file['layers/'+layer].create_dataset('corr_mtx', data=corr_mtx)
            file['layers/'+layer].create_dataset('imgnet_corr_mtx', data=imgnet_corr_mtx)
            file['layers/'+layer].create_dataset('loadings', data=loadings)
            file['layers/'+layer].create_dataset('dictionary', data=dictionary)
            file['layers/'+layer].create_dataset('data_agg', data=data_agg)
            file['layers/'+layer].create_dataset('reconstructed_agg', data=reconstructed_agg)
            
            file['layers/'+layer].attrs['r2'] = r2
            
        file.close()  
        

print('All sweeps processed.')
print('-----------------------------------')