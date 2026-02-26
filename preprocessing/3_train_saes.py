import torch

from bscope import r2_score
import os
import bscope
from IPython import embed
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import h5py as h5
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import bscope.ic as bic



SEMANTIC_JSON_PATH = '/data/hierarchy_metadata/pruned_hierarchy.json'
directory= '/data/h5s/act_normgrad_top_1_False_resnet50/'
# directory= '/data/h5s/int_grad_top_1_False_resnet50_steps_10/'
TYPES = [['contributions', 'positive'], ['activations', 'positive']]

DEVICE = 'cuda:0'

RUN = 'gimmel' # Identifier for this run, used in naming the saved SAEs

for pair in TYPES:

    DATA_TYPE = pair[0]
    SPATIAL_OPERATION = pair[1]

    print('____________________')
    print('Processing directory:', os.path.basename(directory))
    print('____________________')

    file = h5.File(os.path.join(directory, 'data.h5'), 'r')



    # Regularization
    CODE_L1 = None
    ATOM_L1 = 5e-5

    # Training parameters
    BATCH_SIZE = 128
    N = 3 # Dictionary size = N * number of features
    THRESHOLD = 0.8
    LEARNING_RATE = 5e-5
    EPOCHS = 300
    MLP_H = None


    # Data parameters
    DATA_PATH = os.path.join(directory, 'data.h5')
    SAE_PATH = os.path.join(directory, 'saes')




    ID='{}_{}_{}'.format(RUN, DATA_TYPE, SPATIAL_OPERATION)

    SAE_PATH = os.path.join(SAE_PATH, ID)
    os.makedirs(SAE_PATH, exist_ok=True)


    if not os.path.exists(SAE_PATH):
        os.makedirs(SAE_PATH)

    data = h5.File(DATA_PATH, 'r')
    num_layers = len(data.keys())-1
    print(num_layers-1)
    for LAYER_IDX in range(num_layers)[::-1]:
        print('++++++++++')
        print(f'Processing layer {LAYER_IDX}...')
        print('++++++++++')
        LAYER_IDX = str(LAYER_IDX)
        data, targets = bic.load_contribution_data(DATA_PATH, DATA_TYPE, LAYER_IDX, SPATIAL_OPERATION, norm=True)
        dataset = TensorDataset(torch.from_numpy(data).float())

        mask_matrix, mask_labels = bic.get_masks(SEMANTIC_JSON_PATH, leaf_only=True, targets=targets)

        if MLP_H is None:
            MLP_H = data.shape[-1]

        sae = bscope.STSAE(data.shape[-1],
                num_atoms=data.shape[-1]*N,
                threshold=THRESHOLD,
                mlp_hidden_dim=MLP_H).to(DEVICE)


        train_dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=True,
                                                 pin_memory=True,
                                                 num_workers=10)

        eval_dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=10)

        optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss()
        

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)


        for epoch in range(EPOCHS):

            sae.train()
            loss_agg = 0
            for train_batch in train_dataloader:
                train_batch = train_batch[0].to(DEVICE)
                optimizer.zero_grad()

                pre_codes, codes, reconstructed = sae(train_batch)

                loss = loss_fn(reconstructed, train_batch)

                if CODE_L1 is not None:
                    loss += CODE_L1 * torch.mean(torch.abs(codes))
                if ATOM_L1 is not None:
                    loss += ATOM_L1 * torch.mean(torch.abs(sae.dictionary.get_dictionary()))

                loss.backward()
                loss_agg += loss.item()
                optimizer.step()

            losses = loss_agg / len(train_dataloader)
            scheduler.step(losses)
            
            learning_rate = optimizer.param_groups[0]['lr']
            print(f'--Epoch {epoch+1}/{EPOCHS}, Loss: {losses:.4f}, Learning Rate: {learning_rate:.6f}')
            # scheduler.step()
            
            if epoch % 50 == 0:
                sae.eval()

                eval_data_agg = []
                eval_reconstructed_agg = []
                codes_agg = []

                for eval_batch in tqdm.tqdm(eval_dataloader):
                    eval_batch = eval_batch[0].to(DEVICE)

                    pre_codes, codes, reconstructed = sae(eval_batch)
                    eval_data_agg.append(eval_batch.cpu().detach().numpy())
                    eval_reconstructed_agg.append(reconstructed.cpu().detach().numpy())

                    codes_agg.append(codes.cpu().detach().numpy())

                eval_data_agg = np.concatenate(eval_data_agg, axis=0)
                eval_reconstructed_agg = np.concatenate(eval_reconstructed_agg, axis=0)

                eval_data_agg = torch.from_numpy(eval_data_agg).float()
                eval_reconstructed_agg = torch.from_numpy(eval_reconstructed_agg).float()

                r2 = r2_score(eval_data_agg, eval_reconstructed_agg)
                print(f'     Epoch {epoch+1}/{EPOCHS}, R2 Score: {r2:.4f}')
                
                codes = np.concatenate(codes_agg, axis=0)


                firings = np.sum(codes,0)
                alive = firings > 0

                alive_codes = codes[:, alive]
                dictionary = sae.dictionary.get_dictionary().detach().cpu().numpy()[alive]

                print('     Alive codes: ', dictionary.shape[0])
                
                corr = bscope.mtx_corr(mask_matrix.T, alive_codes)
                corr[np.isnan(corr)] = 0

                high_corr = corr > 0.3

                hcs = high_corr.sum(0) > 0
                print('     High correlating codes: ', np.sum(hcs))

                hfs = high_corr.sum(1) > 0
                print('     High correlating features: ', np.sum(hfs))

        torch.save(sae, os.path.join(SAE_PATH, f'layer_{LAYER_IDX}.pt'))
        print(f'Saved SAE for layer {LAYER_IDX} to {os.path.join(SAE_PATH, f"layer_{LAYER_IDX}.pt")}')
                    
