import numpy as np
import bscope
from skkm import Figaro
import pickle
import scipy.stats as stats
from scipy import integrate
import matplotlib.pyplot as plt

# F = Figaro('ppresrve', extension='svg')
F = Figaro('pyy', extension='svg')

topk=1

if topk==1:
    TOPK = 'top1'
else:
    TOPK = 'top5'
# Single run
runtargs = []
runofftargs = []

# for RESULTS_FILE in ['/home/jbmelander/preserve_rn50_pos_cont.pkl', '/home/jbmelander/preserve_rn50_pos_act.pkl']:
labels = ['Cont', 'Act']
ddiffs = []
for j, RESULTS_FILE in enumerate(['/home/jbmelander/preserve_rn50_pos_cont.pkl', '/home/jbmelander/preserve_rn50_pos_act.pkl']):

    with open(RESULTS_FILE, 'rb') as f:
        perturbation_data = pickle.load(f)

    # Extract layers and percentages from the data
    LAYERS = list(perturbation_data.keys())
    PCTS = list(perturbation_data[LAYERS[0]].keys())

    print(LAYERS)

    differences = []
    targs = []
    offtargs = []

    for i, layer in enumerate(LAYERS):
        performance_ratios_mean = []
        performance_ratios_sem = []

        offtarget_performance_ratios_mean = []
        offtarget_performance_ratios_sem = []
        
        for pct in PCTS:
            # Extract original and perturbed accuracies for this layer and percentage
            og_accs = np.array(perturbation_data[layer][pct]['og_{}'.format(TOPK)])
            pert_accs = np.array(perturbation_data[layer][pct]['pert_{}'.format(TOPK)])
            
            # Calculate performance ratio (perturbed / original) for target class
            # Assuming target class is the first class (index 0)
            target_og = og_accs[:, 0]  # Original accuracy for target class
            target_pert = pert_accs[:, 0]  # Perturbed accuracy for target class

            offtarget_og = og_accs[:, 1]  # Original accuracy for off-target classes
            offtarget_pert = pert_accs[:, 1]

            print(target_og)
            print(target_pert)
            
            # Calculate fraction of target class performance (pert/og)
            performance_ratios = target_pert / target_og
            ot_performance_ratios = offtarget_pert / offtarget_og
            
            # Calculate mean and SEM
            mean_ratio = np.nanmean(performance_ratios)
            sem_ratio = stats.sem(performance_ratios)

            offtarget_mean_ratio = np.nanmean(ot_performance_ratios)
            offtarget_sem_ratio = stats.sem(ot_performance_ratios)
            
            performance_ratios_mean.append(mean_ratio)
            performance_ratios_sem.append(sem_ratio)

            offtarget_performance_ratios_mean.append(offtarget_mean_ratio)
            offtarget_performance_ratios_sem.append(offtarget_sem_ratio)
        
        if labels[j]=='Act':
            color = 'blue'
        else:
            color = 'black'
        plt.plot(PCTS, performance_ratios_mean, label='Target', color=color)
        plt.fill_between(PCTS, np.array(performance_ratios_mean)-np.array(performance_ratios_sem), np.array(performance_ratios_mean)+np.array(performance_ratios_sem), color=color, alpha=0.5)
        plt.plot(PCTS, offtarget_performance_ratios_mean, label='Offtarget', color='r')
        plt.fill_between(PCTS, np.array(offtarget_performance_ratios_mean)-np.array(offtarget_performance_ratios_sem), np.array(offtarget_performance_ratios_mean)+np.array(offtarget_performance_ratios_sem), color='r', alpha=0.5)
        F.QQ()
        F.save('layer_{}_{}'.format(layer, labels[j]))
        
        # if 'preserve' in RESULTS_FILE:
        #     performance_ratios_mean = 1 - np.array(performance_ratios_mean)
        #     offtarget_performance_ratios_mean = 1 - np.array(offtarget_performance_ratios_mean)
        targ_auc = bscope.compute_auc(PCTS, performance_ratios_mean)
        offtarg_auc = bscope.compute_auc(PCTS, offtarget_performance_ratios_mean)

        targs.append(targ_auc)
        offtargs.append(offtarg_auc)

        d = (targ_auc - offtarg_auc)/offtarg_auc
        differences.append(d)
    ddiffs.append(differences)

plt.plot(LAYERS, ddiffs[0], label='Cont', color='k')
plt.plot(LAYERS, ddiffs[1], label='Act', color='b')
plt.xlim(0, 16)
plt.xticks([0, 4, 8, 12, 15])
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.ylabel('AUC(Offtarget) - AUC(Target)')
plt.xlabel('Layer')
plt.legend()
F.XX(1/7, 1/4)
F.save('auc_diff_{}.svg'.format(TOPK))



    # plt.plot(LAYERS, differences)
    # runtargs.append(targs)
    # runofftargs.append(offtargs)

# plt.xlim(0, 16)
# plt.xticks([0, 4, 8, 12, 15])
# F.QT()
# F.save('auc_diff_{}.svg'.format(TOPK))

# plt.plot(LAYERS, runtargs[0], label='Cont', color='k')
# plt.plot(LAYERS, runtargs[1], label='Act', color='b')
# plt.plot(LAYERS, runofftargs[0], label='Cont Off', color='k', alpha=0.5)
# plt.plot(LAYERS, runofftargs[1], label='Act Off', color='b', alpha=0.5)
# plt.xlim(0, 16)
# plt.xticks([0, 4, 8, 12, 15])
# plt.legend()
# F.QT()
# F.save('auc_target_{}.svg'.format(TOPK))

    #     # Plot with error bars
