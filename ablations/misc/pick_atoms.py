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
PCT = 2
CLASS = 75
NUM_MODES=2
import skkm
F = skkm.Figaro('FFFF6')

MS_PATH = '/data/h5s/act_normgrad_top_1_False_resnet50/saes/aleph_contributions_positive/mode_summary.h5'


LAYERS = [13,14,15]
WHICH_MODE = 1
PRESERVE = False


torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = 'cuda:0'

model, dataset, dataloader, layers = bscope.ic.get_model(
    imagenet_path='/data/imagenet',
    which_model='resnet50',
    return_layers=True, 
    batch_size=32,
    pin_memory=True,
    num_workers=12,
    device=DEVICE,
    shuffle=False)# Initial small load just to get model

model.eval()

mode_summary = bic.ModeSummary(MS_PATH)

top1, _ = bic.calculate_class_accuracy(model, dataloader, device=DEVICE, target_topk=1, nontarget_topk=1)
top5, _ = bic.calculate_class_accuracy(model, dataloader, device=DEVICE, target_topk=5, nontarget_topk=5)

disruptors = []
for li in LAYERS:
    chs = []
    for WHICH_MODE in range(NUM_MODES):
        idx, atom, loadings = bic.get_top_mode(mode_summary, li, class_idx=CLASS, which_mode=WHICH_MODE) 
        
        N = atom.shape[0] * PCT// 100
        N = int(N)
        channels = list(bic.top_n(atom, N)[0].astype(int))

        if PRESERVE:
            pert_channels = list(set(range(atom.shape[0])) - set(channels))
        else:
            pert_channels = channels

        chs.extend(pert_channels)

    chs = list(set(chs))
    print(len(chs))

    disruptor = bscope.Disruptor(layers[li], pert_channels)
    disruptors.append(disruptor)

for disruptor in disruptors:
    disruptor.activate()

pt1, _ = bic.calculate_class_accuracy(model, dataloader, device=DEVICE, target_topk=1, nontarget_topk=1)
pt5, _ = bic.calculate_class_accuracy(model, dataloader, device=DEVICE, target_topk=5, nontarget_topk=5)

# plt.plot(top1, 'k-', label='Original Top 1')
# plt.plot(pt1, 'r-', label='Perturbed Top 1')
# plt.ylim(0,100)
# plt.legend()
# plt.show()


plt.plot(pt1/top1, 'b-', label='Top 1 Ratio')
# plt.ylim(0,1)
# plt.ylim(0,1)
plt.axvline(x=75, color='r', linestyle='--')
F.XX(1/7, 1/2)
F.save('{}t1'.format(PRESERVE))

plt.plot(pt5/top5, 'k-', label='Original Top 5')
# plt.ylim(0,1)
plt.axvline(x=75, color='r', linestyle='--')

plt.ylim(-0.1, 1.1)
plt.yticks([0, 1])
F.XX(1/7, 1/2)
F.save('{}t5'.format(PRESERVE))


