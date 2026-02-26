import bscope 
import bscope.ic as bic
import os
import tqdm
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import bscope.ic
from IPython import embed

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

BASE_DIR = '/data/h5s/'

STEPS = 25

# Define the contribution types, targets, and whether to use softmax
# The script will loop through all combinations of these and save the results

contribution_types = ['act_normgrad']
contribution_targets = ['entropy']
use_softmax = [True]
# model_type = 'vit'
model_type = 'resnet50'

for ct in contribution_types:
    for target in contribution_targets:
        for softmax in use_softmax:
            print('Running: {}_{}_{}'.format(ct, target, softmax))

            model, dataset, dataloader, layers = bscope.ic.get_model(model_type,
                                                                     return_layers=True,
                                                                batch_size=32,
                                                                num_workers=5,
                                                                pin_memory=False,
                                                              imagenet_path='/data/imagenet')


            # Set the device and move the model to it
            device = 'cuda:0'
            model.to(device)
            model.eval()

            # Create the scope for the model
            scope = bscope.Scope(model, layers)

            # Set the contribution type
            if ct == 'act_normgrad':
                scope.use_act_normgrad()

            elif ct == 'act_grad':
                scope.use_act_grad()

            elif ct == 'int_grad':
                scope.use_int_grad(steps=STEPS)


            # Set the target for the contributions
            if target == 'top_1':
                if softmax:
                    scope.wrt_topk(k=1, softmax=True)
                else:
                    scope.wrt_topk(k=1, softmax=False)

            elif target == 'top_5':
                if softmax:
                    scope.wrt_topk(k=5, softmax=True)
                else:
                    scope.wrt_topk(k=5, softmax=False)

            elif target == 'top_20':
                if softmax:
                    scope.wrt_topk(k=20, softmax=True)
                else:
                    scope.wrt_topk(k=20, softmax=False)

            elif target == 'top_1000':
                if softmax:
                    scope.wrt_topk(k=1000, softmax=True)
                else:
                    scope.wrt_topk(k=1000, softmax=False)

            elif target == 'entropy':
                if softmax:
                    scope.wrt_entropy(softmax=True)
                else:
                    scope.wrt_entropy(softmax=False)

            if model_type == 'vit':
                scope.log_start(reduction=['patch_ei_split', 'patch_sum'])
            elif model_type == 'resnet50':
                scope.log_start(reduction=['ei_split', 'spatial_sum'])

            # Create an identifier for the dataset

            # This identifier is used to save the data
            IDENTIFIER = '{}_{}_{}_{}'.format(ct, target, softmax, model_type)

            if ct == 'int_grad':
                IDENTIFIER += '_steps_{}'.format(STEPS)

            BASE_DIR = '/data/h5s/'
            fname = os.path.join(BASE_DIR, IDENTIFIER)

            os.makedirs(os.path.dirname(fname), exist_ok=True)
            fname = os.path.join(fname, 'data.h5')


            targets = []
            count = 0

            for x, y in tqdm.tqdm(dataloader):
                scope(x)
                targets.append(y.numpy())
                print(scope.contributions[-1].max(), scope.contributions[-1].min(),
                      scope.contributions[-1].mean())


            targets = np.concatenate(targets, 0)

            scope.log_stop()

            with h5.File(fname, 'w') as f:
                print('Saving')
                f.create_dataset('targets', data=targets)
                for i, layer in enumerate(layers):
                    print('--- ', i)
                    activations = scope.log_activations[i]
                    n_chans = activations.shape[1]

                    pos_activations = activations[:, n_chans // 2:]
                    neg_activations = activations[:, :n_chans // 2]
                    pos_gradients = scope.log_gradients[i][:, n_chans // 2:]
                    neg_gradients = scope.log_gradients[i][:, :n_chans // 2]
                    pos_contributions = scope.log_contributions[i][:, n_chans // 2:]
                    neg_contributions = scope.log_contributions[i][:, :n_chans // 2]

                    f.create_group(str(i))

                    f[str(i)].create_group('activations')
                    f[str(i)].create_group('gradients')
                    f[str(i)].create_group('contributions')

                    f[str(i)]['activations'].create_dataset('positive', data=pos_activations)
                    f[str(i)]['activations'].create_dataset('negative', data=neg_activations)

                    f[str(i)]['gradients'].create_dataset('positive', data=pos_gradients)
                    f[str(i)]['gradients'].create_dataset('negative', data=neg_gradients)

                    f[str(i)]['contributions'].create_dataset('positive',
                                                              data=pos_contributions)
                    f[str(i)]['contributions'].create_dataset('negative',
                                                              data=neg_contributions)
