import bscope.ic as bic
import pickle
import json
import bscope
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

SEED = 1992

N_SUBSAMPLE= 10 # The number of samples from each subclass to use for accuracy calculations
N_TRIALS= 60

cont_algo_str = "int_grad_top_1_True_steps_10"
model_type = 'vit'
layer_type = 'mlp'
probe_type = 'contributions' # 'activations' or 'contributions'
spatial_operation = 'positive'  # 'sum', 'positive', 'negative'
PRESERVE = False
DEVICE = 'cuda:2'
MS_PATH = f'/data/codec/h5s/int_grad_top_1_True_{model_type}_{layer_type}_steps_10/saes/aleph_{probe_type}_{spatial_operation}/mode_summary.h5'

if 'contributions' in probe_type:
    probe_short = 'cont'
else:
    probe_short = 'act'

if 'positive' in spatial_operation:
    spatial_short = 'pos'
elif 'negative' in spatial_operation:
    spatial_short = 'neg'
else:
    spatial_short = 'sum'

perturbation_type = 'ablate'  # 'preserve' or 'ablate'
if PRESERVE:
    perturbation_type = 'preserve'
RESULTS_FILE = f'/data/codec/{model_type}_perturbations/{cont_algo_str}_{perturbation_type}_{model_type}_{layer_type}_{spatial_short}_{probe_short}.pkl'

# Create directory if it doesn't exist
import os
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

#LAYERS = [3,6, 7, 9, 11,13,14,15]
#LAYERS = [1]
#LAYERS = [2, 6, 10, 11]
LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
LAYERS = LAYERS[::-1]
PCTS = [1, 2, 5, 10, 18, 25, 50, 80, 99]
#PCTS = [10]
#PCTS = [10, 40, 80]


torch.manual_seed(SEED)
np.random.seed(SEED)


layer_type = None
style = None # Ablation style
if 'vit' in MS_PATH.lower():
    if 'block' in MS_PATH.lower():
        layer_type = 'block'
        style = 'patch_destroy'
    elif 'mlp' in MS_PATH.lower():
        layer_type = 'mlp'
        style = 'mlp_destroy'
    elif 'attention' in MS_PATH.lower():
        layer_type = 'attention'
        style = 'attn_destroy'  # Ablate individual attention channels
    elif 'attn_heads' in MS_PATH.lower():
        layer_type = 'attn_heads'
        style = 'attn_head_destroy' # Ablate entire heads

model, dataset, dataloader, layers = bscope.ic.get_model(
    imagenet_path='/data/imagenet',
    which_model=model_type,
    return_layers=True, 
    batch_size=128,
    pin_memory=False,
    device=DEVICE,
    shuffle=False,
    layer_type=layer_type)# Initial small load just to get model


del dataset, dataloader 
model.eval()

mode_summary = bic.ModeSummary(MS_PATH)

perturbation_data = {}
for li in LAYERS:
    perturbation_data[li] = {}
    for pct in PCTS:
        perturbation_data[li][pct] = {}
        perturbation_data[li][pct]['og_top1'] = []
        perturbation_data[li][pct]['og_top5'] = []
        perturbation_data[li][pct]['pert_top1'] = []
        perturbation_data[li][pct]['pert_top5'] = []
        perturbation_data[li][pct]['subclasses'] = []
        perturbation_data[li][pct]['atom'] = []

        for trial in range(N_TRIALS):
            subclasses= [np.random.randint(0,1000), np.random.randint(0,1000)]
            print('Subclasses ', subclasses)
            dataloader = bic.get_model(model_type, return_layers=False, imagenet_path='/data/imagenet', device=DEVICE,subsample=N_SUBSAMPLE,subclasses=subclasses,dataloader_only=True, batch_size=50)

            # Get the top mode
            idx, atom, loadings, _ = bic.get_top_mode(mode_summary, li, subclasses[0], 1)
            print(atom.shape)
            
            if 'entropy' in MS_PATH.lower():
                if 'contributions' in MS_PATH.lower():
                    atom = atom * -1

            if 'concat' in MS_PATH.lower():
                n_chans = atom.shape[0] // 2
                atom = atom[:n_chans]

            num_chans = atom.shape[0]
            num_to_keep = int(num_chans * pct / 100)

            # Get the indices of the channels to keepb
            channels = list(bic.top_n(atom, num_to_keep)[0].astype(int))

            if PRESERVE:
                pert_channels = list(set(range(num_chans)) - set(channels))
            else:
                pert_channels = channels

            
            # Create disruptor
            top1, top5 = bic.calculate_subsample_accuracy(model, dataloader, subclasses, device=DEVICE)

            print(top1)
            if 'vit' in MS_PATH.lower():
                disruptor = bscope.Disruptor(layers[li], pert_channels, style=style)
                disruptor.activate(heads = model.blocks[0].attn.num_heads)
            else:
                disruptor = bscope.Disruptor(layers[li], pert_channels)
                disruptor.activate()
            pt1, pt5 = bic.calculate_subsample_accuracy(model, dataloader, subclasses, device=DEVICE)
            disruptor.deactivate()

            print('--- original: ',top1, top5)
            print('--- perturbed: ',pt1, pt5)

            perturbation_data[li][pct]['og_top1'].append(list(top1))
            perturbation_data[li][pct]['og_top5'].append(list(top5))
            perturbation_data[li][pct]['pert_top1'].append(list(pt1))
            perturbation_data[li][pct]['pert_top5'].append(list(pt5))

            perturbation_data[li][pct]['subclasses'].append(subclasses)
            perturbation_data[li][pct]['atom'].append(idx)



# Save (handles all Python types including numpy)
with open(RESULTS_FILE, 'wb') as f:
    pickle.dump(perturbation_data, f)



