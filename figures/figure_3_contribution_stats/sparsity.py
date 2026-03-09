import bscope.ic as bic
import bscope
import numpy as np
import matplotlib.pyplot as plt
import sys
from fycus import Fycus

FIG = 'figure_3'

F = Fycus(FIG)

path = '/data/h5s/int_grad_top_1_False_resnet50_steps_10/data.h5'


act_hoyers = []
con_hoyers = []

act_hoyers_sem = []
con_hoyers_sem = []

bcon_hoyers = []
bact_hoyers = []

for l in range(16):
    data = bic.load_contribution_data(path, 'contributions', l, 'sum')[0]
    act_data = bic.load_contribution_data(path, 'activations', l, 'positive')[0]

    A =bscope.hoyer(act_data)
    C = bscope.hoyer(data)
    
    bcon_hoyers.append(C)
    bact_hoyers.append(A)
    print(data.shape)
    print(A.shape)


        # Make the bins increments of 0.05 from 0 to 1
    bins = np.arange(0, 1.05, 0.025)

    plt.hist(A, bins=bins, label='activations', color='b', alpha=0.9)
    plt.hist(C, bins=bins, label='contributions', color='k', alpha=0.9)
    plt.xlabel('Sparsity (hoyer)')
    plt.ylabel('Count')
    F.QT()
    F.save('B_layer_{}_sparsity_hist'.format(l))

    act_hoyers.append(np.nanmean(A))
    con_hoyers.append(np.nanmean(C))

    act_hoyers_sem.append(np.nanstd(A))
    con_hoyers_sem.append(np.nanstd(C))



plt.errorbar(np.arange(16), con_hoyers, yerr=con_hoyers_sem, fmt='k')
plt.errorbar(np.arange(16), act_hoyers, yerr=act_hoyers_sem, fmt='b')
plt.title('Average Sparsity (Hoyer) per Layer')

plt.legend(['contributions', 'activations'])
plt.ylabel('avg sparsity (hoyer)')
plt.xlabel('layer')
F.QT()
F.save('B_avg_sparsity_per_layer')

# print(len(con_hoyers))
# print(np.arange(16).shape)
# Make bcon black and bact blue
plt.boxplot(bcon_hoyers, positions=np.arange(16)-0.1, showfliers=False, boxprops=dict(color='k'), medianprops=dict(color='k'))
plt.boxplot(bact_hoyers, positions=np.arange(16)+0.1, showfliers=False, boxprops=dict(color='b'), medianprops=dict(color='b'))
plt.xticks([0, 4, 8, 12, 15], ['0', '4', '8', '12', '15'])
plt.ylim(0, 0.8)
F.QT()
F.save('B_boxplot_sparsity_per_layer')
