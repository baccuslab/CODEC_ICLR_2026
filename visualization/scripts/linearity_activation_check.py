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


filter_size = 2
contrast = 6
LAYER = 12
# CLASS = 486 #cello
CLASS = 889 #violin
# CLASS = 642 # marimba
device = 'cuda:0'
NCHAN = 1
CHANNELS = NCHAN
use_norm = False

F = fycus.Fycus('normed_channels_{}_filt_{}_contrast_{}_layer_{}_class_{}'.format(NCHAN, filter_size, contrast, LAYER, CLASS))
F.XX(2,2)


def sign_split(M):
    positive = np.copy(M)
    negative = np.copy(M)
    abs_M = np.abs(M)
    positive[positive<0] = 0
    negative[negative>0] = 0
    negative = np.abs(negative)
    return abs_M, positive, negative

# -------------------------------------------------------------------------
model, dataset, dataloader, layers = bscope.ic.get_model('resnet50', return_layers=True, imagenet_path='/data/imagenet/')
model.eval()
model.to(device)
inspector = bscope.Inspector([layers[LAYER]], to_numpy=False)
DEVICE = next(model.parameters()).device

# 4. MAIN SCRIPT
# =============================================================================

img_idx = 18202 

WHICH_MODE = 0

mode_summary_path = '/data/h5s/int_grad_top_1_False_resnet50_steps_10/saes/aleph_contributions_positive/mode_summary.h5'

mode_summary = bic.ModeSummary(mode_summary_path)
mode_idx, atom, mode_loadings, corr = bic.get_top_mode(mode_summary, LAYER, CLASS, WHICH_MODE)

# Normalize channel weights by the max atom value among the top channels
chan_idxs, vals = bic.top_n(atom, CHANNELS)
top_channels = list(np.sort(chan_idxs))

# 4b. Setup model
model.zero_grad()




        # 4c. Forward pass
        # ---------------------------------------------------------------------
SPECIFIC_CLASS_IDX = img_idx // 50

X = dataset[img_idx][0].unsqueeze(0).to(DEVICE)
X.requires_grad_(True)

Y = model(X)
Y = torch.nn.Softmax(dim=1)(Y)

activations = inspector.activations[0]

Jy = torch.autograd.grad(Y[:,SPECIFIC_CLASS_IDX].sum(), activations, retain_graph=True)[0].cpu().detach().numpy()

# From dataloader preprocessing, the images are normalized by these mean and std values. To visualize the image properly, we need to unnormalize it using these values.
_mean=[0.485, 0.456, 0.406]
_std=[0.229, 0.224, 0.225]

torch_mean = torch.tensor(_mean).view(1,3,1,1).to(DEVICE)
torch_std = torch.tensor(_std).view(1,3,1,1).to(DEVICE)

mode_activations = activations[:, top_channels, :, :]

# Randomly shuffle the top channels indices
shuffled_indices = np.random.permutation(len(top_channels))
shuf_mode_activations = mode_activations[:, shuffled_indices, :, :]

assert torch.allclose(mode_activations[:, shuffled_indices, :, :], shuf_mode_activations), "Shuffling the channels should not change the activations"


mode_Jy = Jy[:, top_channels, :, :]
shuffled_mode_Jy = mode_Jy[:, shuffled_indices, :, :]

assert np.allclose(mode_Jy[:, shuffled_indices, :, :], shuffled_mode_Jy), "Shuffling the channels should not change the Jy values"

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

z = []
z_approx = []
for _k in tqdm.tqdm(range(NCHAN)):
    # Weight for this channel: atom value normalized by max atom value
    channel_weight = 1.0
    for _h in range(h):
        for _w in range(w):
            _Jz = torch.autograd.grad(mode_activations[0,_k,_h,_w], X, retain_graph=True)[0].cpu().detach().numpy()
            _Jy = mode_Jy[:,_k,_h,_w]
            z.append(mode_activations[0,_k,_h,_w].item())

            J = np.einsum('abcd,a->abcd', _Jz, _Jy)

            linear_activation = np.einsum('abcd,abcd->a', _Jz, X.cpu().detach().numpy())
            z_approx.append(linear_activation)

            J_norm = np.linalg.norm(J) + 1e-16
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
                PossumJyJz += J * channel_weight
                PossumXJyJz += np.einsum('abcd,abcd->abcd', J, X.cpu().detach().numpy()) * channel_weight
                PossumJyJzNorm += _JyJzNorm * channel_weight
                PossumXJyJzNorm += np.einsum('abcd,abcd->abcd', _JyJzNorm, X.cpu().detach().numpy()) * channel_weight
            else:
                NegsumJyJz += J * channel_weight
                NegsumXJyJz += np.einsum('abcd,abcd->abcd', J, X.cpu().detach().numpy()) * channel_weight
                NegsumJyJzNorm += _JyJzNorm * channel_weight
                NegsumXJyJzNorm += np.einsum('abcd,abcd->abcd', _JyJzNorm, X.cpu().detach().numpy()) * channel_weight

            Jz[_k, _h, _w, :, :, :] = _Jz * channel_weight
            Jy_arr[_k, _h, _w] = _Jy * channel_weight
            JyJz[_k, _h, _w, :, :, :] = J *channel_weight

z = np.array(z)
z_approx = np.array(z_approx)
plt.scatter(z, z_approx)
plt.xlabel('Actual activation value')
plt.ylabel('Linear approximation of activation value')
plt.title('Actual vs Linear Approximation of Activation Value')
plt.show()


JyJz = JyJz.sum(axis=(0,1,2)).transpose(1,2,0)
JyJzNorm = JyJzNorm.sum(axis=(0,1,2)).transpose(1,2,0)

PosJyJz = PossumJyJz[0].transpose(1,2,0)
NegJyJz = NegsumJyJz[0].transpose(1,2,0)
PosJyJzNorm = PossumJyJzNorm[0].transpose(1,2,0)
NegJyJzNorm = NegsumJyJzNorm[0].transpose(1,2,0)

# 4e. Visualize results
# ---------------------------------------------------------------------
X_np = X.cpu().detach()[0].permute(1,2,0).numpy()

np_std = np.array(_std)[np.newaxis, np.newaxis, :]
np_mean = np.array(_mean)[np.newaxis, np.newaxis, :]
X_vis = X_np * np_std + np_mean
# X_vis = X_np-np.min(X_np)
# X_vis = X_vis / (np.max(X_vis) + 1e-8)

XJyJz = np.einsum('abc,abc->abc', JyJz, X_np)
PosXJyJz = np.einsum('abc,abc->abc', PosJyJz, X_np)
NegXJyJz = np.einsum('abc,abc->abc', NegJyJz, X_np)
PosXJyJzNorm = np.einsum('abc,abc->abc', PosJyJzNorm, X_np)
NegXJyJzNorm = np.einsum('abc,abc->abc', NegJyJzNorm, X_np)
XJyJzNorm = np.einsum('abc,abc->abc', JyJzNorm, X_np)

plt.imshow(X_vis, interpolation='none')
plt.show()

plt.imshow(np.abs(JyJz) / np.max(np.abs(JyJz)), interpolation='none')
plt.show()

plt.imshow(np.abs(XJyJz) / np.max(np.abs(XJyJz)), interpolation='none')
plt.show()

z = np.abs(XJyJz) / np.max(np.abs(XJyJz))
z = np.max(z, axis=2)

z = median_filter(z, size=filter_size)
plt.imshow(z*4,cmap='gray', interpolation='none', clim=(0,1))
plt.show()

# normalize = lambda x: (x) / (np.max(np.abs(x)))

# # if use_norm:
# #     JyJz = JyJzNorm
# #     PosJyJz = PosJyJzNorm
# #     NegJyJz = NegJyJzNorm
# #     XJyJz = XJyJzNorm
# #     PosXJyJz = PosXJyJzNorm
# #     NegXJyJz = NegXJyJzNorm

# fig, ax = plt.subplots(1, 3, figsize=(10,5))

# # XJyJz = XJyJz * _std[..., np.newaxis, np.newaxis] + _mean[..., np.newaxis, np.newaxis]
# # This doesn't work, add axes to _std and _mean to match the shape of XJyJz

# # std = np.array(_std)[np.newaxis, np.newaxis, :]
# # mean = np.array(_mean)[np.newaxis, np.newaxis, :]

# mask, _, _ = sign_split(XJyJz)
# if filter_size > 1:
#     mask = [median_filter(mask[:,:,c], size=filter_size) for c in range(3)]
#     mask = np.stack(mask, axis=2)
# mask = mask.mean(axis=2)
# # mask = median_filter(mask, size=filter_size)[:,:,np.newaxis]

# # [:,:,np.newaxis]



# # mask = mask.max(axis=2)[:,:,np.newaxis]

# mask -= np.min(mask)
# mask = mask / (np.max(mask))

# mask *= contrast

# mask = np.clip(mask, 0, 1)

# masked_image = X_vis * mask[:,:,np.newaxis]

# fig, ax = plt.subplots(1, 3, figsize=(10,5))
# ax[0].imshow(X_vis, interpolation='none')
# ax[0].set_title('Original Image')
# ax[1].imshow(masked_image, interpolation='none')
# ax[1].set_title(f'Masked Image (Mode {WHICH_MODE})')
# ax[2].imshow(mask, cmap='gray', interpolation='none', clim=(0,1))
# F.XX(1,2)
# F.save(f'{img_idx}_mode_{WHICH_MODE}', dpi=150)
