import h5py as h5
import numpy as np
import bscope
import bscope.ic as bic
import glob
import os


h5_keywords = ['act_normgrad', 'False']
sae_keywords = ['aleph', 'sum']


top_level_directories = glob.glob('/data/h5s/*')
top_level_directories = [d for d in top_level_directories if all(keyword in os.path.basename(d) for keyword in h5_keywords)]


for directory in top_level_directories:
    print('Searching {}'.format(os.path.basename(directory)))
    sae_directories = glob.glob(os.path.join(directory, 'saes', '*'))
    sae_directories = [d for d in sae_directories if all(keyword in os.path.basename(d) for keyword in sae_keywords)]


    if len(sae_directories) == 0:
        print('!! No SAE directories found in:', os.path.basename(directory))
        continue

    datapath = os.path.join(directory, 'data.h5')
    print('Found SAE directories:')
    [print('>', os.path.basename(d)) for d in sae_directories]


    for saepath in sae_directories:
        try:
            if os.path.exists(saepath):
                print('+++Found SAE path:', saepath)
            else:
                print('+++Didnt find SAE path:', saepath)
                input('+++ERROR')

            mode_summary_path = os.path.join(saepath, 'mode_summary.h5')
            if os.path.exists(mode_summary_path):
                print('+++Mode summary already exists:', mode_summary_path)
                print('+++Recomputing mode summary...')
            else:
                print('+++Creating new mode summary:', mode_summary_path)


            file = h5.File(mode_summary_path, 'w')
            device= 'cuda:0'

            SYN_COMPUTED=False
            layers = []
            file.create_group('layers')
            for single_sae_path in glob.glob(os.path.join(saepath, '*.pt')):
                if 'contributions' in single_sae_path:
                    datatype = 'contributions'
                if 'activations' in single_sae_path:
                    datatype = 'activations'
                if 'gradients' in single_sae_path:
                    datatype = 'gradients'

                if 'sum' in single_sae_path:
                    spatial_operation = 'sum'
                if 'positive' in single_sae_path:
                    spatial_operation = 'positive'
                if 'negative' in single_sae_path:
                    spatial_operation = 'negative'
                if 'concat' in single_sae_path:
                    spatial_operation = 'concat'


                
                print('+++Using single_sae_path with datatype:', single_sae_path, datatype)
                layer = single_sae_path.split('/')[-1].split('_')[-1].split('.')[0]
                file.create_group('layers/' + layer)

                data, targets = bic.load_contribution_data(datapath, datatype, layer, spatial_operation)
                if not SYN_COMPUTED:
                    mask_matrix, mask_labels = bic.get_masks()
                    file.create_dataset('mask_labels', data=mask_labels)
                    file.create_dataset('mask_matrix', data=mask_matrix)

                    print(mask_matrix.shape)
                    imgnet_mask_matrix, imgnet_mask_labels = bic.get_masks(leaf_only=True)
                    file.create_dataset('imgnet_mask_labels', data=imgnet_mask_labels)
                    file.create_dataset('imgnet_mask_matrix', data=imgnet_mask_matrix)
                    print(imgnet_mask_matrix.shape)


                    SYN_COMPUTED = True
                    print('+++Saved mask matrices and labels to mode summary h5')

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
                print('+++R2:', r2 )
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

                # Make r2 an attr
                file['layers/'+layer].attrs['r2'] = r2
        except Exception as e:
            print('+++Error processing SAE path:', saepath)
            print('+++Error message:', e)
            continue
        
