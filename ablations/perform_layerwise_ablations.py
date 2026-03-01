import bscope.ic as bic
import pickle
import json
import bscope
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

SEED = 1992

N_SUBSAMPLE= 10
N_TRIALS= 20

MS_PATH = '/data/h5s/act_normgrad_top_1_False_resnet50/saes/aleph_contributions_sum/mode_summary.h5'

RESULTS_FILE = '/home/jbmelander/preserve_rn50_sum_cont.pkl'
PRESERVE = True

LAYERS = [3,6, 7, 9, 11,13,14,15]
LAYERS = LAYERS[::-1]
PCTS = [1, 2, 5, 10, 18, 25, 50, 80, 99]


torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = 'cuda:0'

model, dataset, dataloader, layers = bscope.ic.get_model(
    imagenet_path='/data/imagenet',
    which_model='resnet50',
    return_layers=True, 
    batch_size=128,
    pin_memory=False,
    device=DEVICE,
    shuffle=False)# Initial small load just to get model

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
            dataloader = bic.get_model('resnet50', return_layers=False, imagenet_path='/data/imagenet', device=DEVICE,subsample=N_SUBSAMPLE,subclasses=subclasses,dataloader_only=True, batch_size=50)

            # Get the top mode
            idx, atom, loadings = bic.get_top_mode(mode_summary, li, subclasses[0], 1)
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
            top1, top5 = bic.calculate_subsample_accuracy(model, dataloader, subclasses)

            print(top1)
            if 'vit' in MS_PATH.lower():
                disruptor = bscope.Disruptor(layers[li], pert_channels, style='patch_destroy')
            else:
                disruptor = bscope.Disruptor(layers[li], pert_channels)
            disruptor.activate()
            pt1, pt5 = bic.calculate_subsample_accuracy(model, dataloader, subclasses)
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



