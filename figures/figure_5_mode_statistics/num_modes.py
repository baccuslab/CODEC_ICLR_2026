import skkm 
import numpy as np
from scipy.stats import pearsonr
import bscope
import scipy.stats
import tqdm
import skkm 
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import bscope.ic as bic
import os
FIG = 'figure_5'

from fycus import Fycus
F = Fycus(FIG, extension='svg')

c_ms_path = '/data/h5s/int_grad_top_1_False_resnet50_steps_10/saes/aleph_contributions_positive/mode_summary.h5'
c_ms = bic.ModeSummary(c_ms_path)

a_ms_path = '/data/h5s/int_grad_top_1_False_resnet50_steps_10/saes/aleph_activations_positive/mode_summary.h5'
a_ms = bic.ModeSummary(a_ms_path)

cns = []
ans = []
n_chans = []
for i in tqdm.tqdm(range(16)):
    n_chan = c_ms.layers[i].dictionary.shape[1]
    cn = c_ms.layers[i].dictionary.shape[0]
    an = a_ms.layers[i].dictionary.shape[0]

    cns.append(cn)
    ans.append(an)
    n_chans.append(n_chan)

cns = np.array(cns)
ans = np.array(ans)
n_chans = np.array(n_chans)

plt.plot(cns, label='Contributions')
plt.plot(ans, label='Activations')
plt.legend()
F.QT()
F.save('num_modes')

plt.plot(cns/n_chans, label='Contributions')
plt.plot(ans/n_chans, label='Activations')
plt.legend()
F.QT()
F.save('num_modes_norm')



