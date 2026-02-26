# 1. IMPORTS
# =============================================================================

from IPython import embed

import bopt
import bscope
import bscope.ic as bic
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from scipy.ndimage import median_filter, gaussian_filter

import fycus

# 2. CONFIGURATION
# =============================================================================


filter_size = 3
contrast = 6
LAYER =  9
CLASS = 889
device = 'cuda:1'
NCHAN = 12
CHANNELS = NCHAN
use_norm = False 

F = fycus.Fycus('fff')
F.XX(2,2)


def sign_split(M):
    positive = np.copy(M)
    negative = np.copy(M)
    abs_M = np.abs(M)
    positive[positive<0] = 0
    negative[negative>0] = 0
    negative = np.abs(negative)
    return abs_M, positive, negative

# 4. MAIN SCRIPT
# =============================================================================

# img_idxs = [22801, 22819, 44475, 44485, 44498, 24314, 24316, 24323, 32117, 32128, 32148]
# img_idxs = [22801, 44453, 44488]#  22819, 44475, 44485, 44498, 24314, 24316, 24323, 32117, 32128, 32148]
img_idxs = [24348, 24326, 44488, 44493, 44453, 44468, 32109, 32128]   #, 22801, 22819, 22818, 44453, 44488, 24314, 24319, 32109, 32128]#  22819, 44475, 44485, 44498, 24314, 24316, 24323, 32117, 32128, 32148]

for img_idx in img_idxs:
    for WHICH_MODE in [1, 2, 3, 4, 5,6,7,8]: #, 5, 6, 7, -1, -2, -3, -4]:

        # mode_summary_path = '/data/h5s/int_grad_top_1_False_resnet50_steps_10/saes/aleph_contributions_positive/mode_summary.h5'
        mode_summary_path = '/data/h5s/act_normgrad_top_1_False_resnet50/saes/aleph_contributions_positive/mode_summary.h5'

        mode_summary = bic.ModeSummary(mode_summary_path)
        mode_idx, atom, mode_loadings, corr = bic.get_top_mode(mode_summary, LAYER, CLASS, WHICH_MODE)
        chan_idxs, vals = bic.top_n(atom, CHANNELS)

        if CHANNELS == 'all':
            chan_idxs = np.arange(atom.shape[0])
            print(f'Using all channels: {len(chan_idxs)}')
        top_channels = list(np.sort(chan_idxs))


        # 4b. Setup model
        # -------------------------------------------------------------------------
        model, dataset, dataloader, layers = bscope.ic.get_model('resnet50', return_layers=True, imagenet_path='/data/imagenet/')
        model.eval()
        model.to(device)
        inspector = bscope.Inspector([layers[LAYER]], to_numpy=False)
        DEVICE = next(model.parameters()).device



        # 4c. Forward pass
        # ---------------------------------------------------------------------
        SPECIFIC_CLASS_IDX = img_idx // 50

        X = dataset[img_idx][0].unsqueeze(0).to(DEVICE)
        X.requires_grad_(True)

        Y = model(X)
        Y = torch.nn.Softmax(dim=1)(Y)

        activations = inspector.activations[0]

        Jy = torch.autograd.grad(Y[:,SPECIFIC_CLASS_IDX].sum(), activations, retain_graph=True)[0].cpu().detach().numpy()


        mode_activations = activations[:, top_channels, :, :]
        mode_Jy = Jy[:, top_channels, :, :]

        print('Mode activations shape:', mode_activations.shape)
        print('Mode Jy shape:', mode_Jy.shape)

        model.zero_grad()
        X.grad = None

        # 4d. Compute contributions
        # ---------------------------------------------------------------------
        h, w = mode_activations.shape[2], mode_activations.shape[3]

        Jz = np.zeros((NCHAN, h, w, 3, 224, 224), dtype='float32')
        Jy_arr = np.zeros((NCHAN, h, w), dtype='float32')
        JyJz = np.zeros((NCHAN, h, w, 3, 224, 224), dtype='float32')
        JyJzNorm = np.zeros((NCHAN, h, w, 3, 224, 224), dtype='float32')
        C = np.zeros((NCHAN, h, w), dtype='float32')

        first = True

        for _k in tqdm.tqdm(range(NCHAN)):
            for _h in range(h):
                for _w in range(w):
                    _Jz = torch.autograd.grad(mode_activations[0,_k,_h,_w], X, retain_graph=True)[0].cpu().detach().numpy()
                    _Jy = mode_Jy[:,_k,_h,_w]

                    J = np.einsum('abcd,a->abcd', _Jz, _Jy)

                    J_norm = np.linalg.norm(J) + 1e-8
                    J_norm = J / J_norm

                    _JyJzNorm = np.einsum('abcd,a->abcd', J_norm, _Jy)
                    JyJzNorm[_k, _h, _w, :, :, :] = _JyJzNorm

                    Jz[_k, _h, _w, :, :, :] = _Jz
                    Jy_arr[_k, _h, _w] = _Jy
                    JyJz[_k, _h, _w, :, :, :] = J

                    model.zero_grad()

                    _C = np.einsum('abcd,abcd->a', J, X.cpu().detach().numpy())
                    C[_k, _h, _w] = _C

                    if first:
                        PossumJyJz = np.zeros_like(J)
                        PossumXJyJz = np.zeros_like(X.cpu().detach().numpy())
                        NegsumJyJz = np.zeros_like(J)
                        NegsumXJyJz = np.zeros_like(X.cpu().detach().numpy())

                        PossumJyJzNorm = np.zeros_like(J)
                        NegsumJyJzNorm = np.zeros_like(J)
                        PossumXJyJzNorm = np.zeros_like(X.cpu().detach().numpy())
                        NegsumXJyJzNorm = np.zeros_like(X.cpu().detach().numpy())
                        first = False

                    if _C > 0:
                        PossumJyJz += J
                        PossumXJyJz += np.einsum('abcd,abcd->abcd', J, X.cpu().detach().numpy())
                        PossumJyJzNorm += _JyJzNorm
                        PossumXJyJzNorm += np.einsum('abcd,abcd->abcd', _JyJzNorm, X.cpu().detach().numpy())
                    else:
                        NegsumJyJz += J
                        NegsumXJyJz += np.einsum('abcd,abcd->abcd', J, X.cpu().detach().numpy())
                        NegsumJyJzNorm += _JyJzNorm
                        NegsumXJyJzNorm += np.einsum('abcd,abcd->abcd', _JyJzNorm, X.cpu().detach().numpy())

        JyJz = JyJz.sum(axis=(0,1,2)).transpose(1,2,0)
        JyJzNorm = JyJzNorm.sum(axis=(0,1,2)).transpose(1,2,0)

        PosJyJz = PossumJyJz[0].transpose(1,2,0)
        NegJyJz = NegsumJyJz[0].transpose(1,2,0)
        PosJyJzNorm = PossumJyJzNorm[0].transpose(1,2,0)
        NegJyJzNorm = NegsumJyJzNorm[0].transpose(1,2,0)

        # 4e. Visualize results
        plt.tight_layout()
        # ---------------------------------------------------------------------
        X_np = X.cpu().detach()[0].permute(1,2,0).numpy()
        X_vis = bic.normalize(X_np)

        XJyJz = np.einsum('abc,abc->abc', JyJz, X_np)
        PosXJyJz = np.einsum('abc,abc->abc', PosJyJz, X_np)
        NegXJyJz = np.einsum('abc,abc->abc', NegJyJz, X_np)
        PosXJyJzNorm = np.einsum('abc,abc->abc', PosJyJzNorm, X_np)
        NegXJyJzNorm = np.einsum('abc,abc->abc', NegJyJzNorm, X_np)
        XJyJzNorm = np.einsum('abc,abc->abc', JyJzNorm, X_np)



        normalize = lambda x: (x) / (np.max(np.abs(x)) + 1e-8)

        # if use_norm:
        #     JyJz = JyJzNorm
        #     PosJyJz = PosJyJzNorm
        #     NegJyJz = NegJyJzNorm
        #     XJyJz = XJyJzNorm 
        #     PosXJyJz = PosXJyJzNorm
        #     NegXJyJz = NegXJyJzNorm

        fig, ax = plt.subplots(1, 2, figsize=(10,5))

        mask, _, _ = sign_split(XJyJz)
        # mask = mask.mean(axis=2)[:,:,np.newaxis]

        if filter_size > 1:
            mask = [median_filter(mask[:,:,c], size=filter_size) for c in range(3)]
            mask = np.stack(mask, axis=2)
            # mask = median_filter(mask, size=filter_size)
        mask = np.max(mask, axis=2)[:,:,np.newaxis]
        mask = mask - np.min(mask)
        mask = mask / (np.max(mask) + 1e-8)

        mask *= contrast

        mask = np.clip(mask, 0, 1) 
        # mask = bic.normalize(mask)
        # mask = mask - np.min(mask)
        # mask = mask / (np.max(mask) + 1e-8)


        masked_image = X_vis * mask

        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        ax[0].imshow(X_vis, interpolation='none')
        ax[0].set_title('Original Image')
        ax[1].imshow(masked_image, interpolation='none')
        ax[1].set_title(f'Masked Image (Mode {WHICH_MODE})')
        F.save(f'{img_idx}_mode_{WHICH_MODE}_combined', dpi=150)

        # mask, _, _ = sign_split(PosXJyJz)
        # # mask = np.abs(PosXJyJz)
        # mask = mask / (np.max(mask) + 1e-8)
        # mask *= contrast

        # if filter_size > 1:
        #     mask = median_filter(mask, size=filter_size)

        # mask = np.clip(mask, 0, 1)
        # mask = np.max(mask, axis=2)[:,:,np.newaxis]
        #                             # Combine across color channels
        # masked_image = X_vis * mask
        # # ax[2,1].imshow(masked_image, interpolation='none')
        # # ax[2,1].set_title('XJyJz + Masked Image')

        # mask, _, _ = sign_split(NegXJyJz)
        # mask = mask / (np.max(mask) + 1e-8)
        # mask *= contrast
        # if filter_size > 1:
        #     mask = median_filter(mask, size=filter_size)
        # mask = np.clip(mask, 0, 1)
        # mask = np.max(mask, axis=2)[:,:,np.newaxis]
        # masked_image = X_vis * mask
        # # ax[2,2].imshow(masked_image, interpolation='none')
        # ax[2,2].set_title('XJyJz - Masked Image')
        # F.save(f'{img_idx}_mode_{WHICH_MODE}.png', dpi=150)


