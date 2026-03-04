import numpy as np
import bscope
import pickle
import scipy.stats as stats
from scipy import integrate
import matplotlib.pyplot as plt
from fycus import Fycus

cont_algo_str = "int_grad_top_1_True_steps_10"
model_types = ['vit']
layer_types = ['block', 'mlp', 'attention']
perturbation_types = ['ablate']  # 'preserve' or 'ablate'
topk_options = [1, 5] # [1, 5]
cont_spatial_short = 'pos'
act_spatial_short = 'sum'

# Meta trials for noise estimation
N_META_TRIALS = 6  # Number of meta trials
TRIALS_PER_META = 10  # Trials within each meta trial
N_TRIALS = N_META_TRIALS * TRIALS_PER_META  # Total should be 60

# Replacement value for inf values in data
INF_REPLACEMENT = 10.0

F = Fycus('ablations', extension='svg')

FIG = f'pyy'

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
for model_type in model_types:
    for layer_type in layer_types:
        for perturbation_type in perturbation_types:
            for topk in topk_options:
                if topk==1:
                    TOPK = 'top1'
                else:
                    TOPK = 'top5'
                FIG = f'{cont_algo_str}_auc_{model_type}_{layer_type}_{perturbation_type}_top{topk}'
                ddiffs = []

                for j, RESULTS_FILE in enumerate([f'/data/codec/{model_type}_perturbations/{cont_algo_str}_{perturbation_type}_{model_type}_{layer_type}_{cont_spatial_short}_cont.pkl', f'/data/codec/{model_type}_perturbations/{cont_algo_str}_{perturbation_type}_{model_type}_{layer_type}_{act_spatial_short}_act.pkl']):

                    with open(RESULTS_FILE, 'rb') as f:
                        perturbation_data = pickle.load(f)

                    # Extract layers and percentages from the data
                    LAYERS = list(perturbation_data.keys())
                    PCTS = list(perturbation_data[LAYERS[0]].keys())

                    print(LAYERS)

                    differences = []  # Will have shape (n_layers, n_meta)
                    targs = []  # Will have shape (n_layers, n_meta)
                    offtargs = []  # Will have shape (n_layers, n_meta)

                    for i, layer in enumerate(LAYERS):
                        performance_ratios_mean = []
                        performance_ratios_sem = []

                        offtarget_performance_ratios_mean = []
                        offtarget_performance_ratios_sem = []
                        
                        # Arrays to store per-meta-trial performance ratios for AUC computation
                        meta_performance_ratios = []  # Will have shape (n_pcts, n_meta)
                        meta_offtarget_performance_ratios = []  # Will have shape (n_pcts, n_meta)
                        
                        for pct in PCTS:
                            # Extract original and perturbed accuracies for this layer and percentage
                            og_accs = np.array(perturbation_data[layer][pct]['og_{}'.format(TOPK)])
                            pert_accs = np.array(perturbation_data[layer][pct]['pert_{}'.format(TOPK)])
                            
                            # Reshape to (n_meta, trials_per_meta, 2) for meta trial analysis
                            n_trials = og_accs.shape[0]
                            og_accs_reshaped = og_accs.reshape(N_META_TRIALS, n_trials // N_META_TRIALS, 2)
                            pert_accs_reshaped = pert_accs.reshape(N_META_TRIALS, n_trials // N_META_TRIALS, 2)
                            
                            # Calculate performance ratio (perturbed / original) for target class
                            # *_accs has shape (n_meta, trials_per_meta, 2) where last dim: 0=target, 1=off-target
                            target_og = og_accs_reshaped[:, :, 0]  # Shape: (n_meta, trials_per_meta)
                            target_pert = pert_accs_reshaped[:, :, 0]

                            offtarget_og = og_accs_reshaped[:, :, 1]
                            offtarget_pert = pert_accs_reshaped[:, :, 1]

                            # print(target_og)
                            # print(target_pert)
                            
                            # Calculate fraction of target class performance (pert/og)
                            # Shape: (n_meta, trials_per_meta)
                            performance_ratios = target_pert / target_og
                            ot_performance_ratios = offtarget_pert / offtarget_og
                            
                            # Replace inf values with INF_REPLACEMENT
                            performance_ratios[np.isinf(performance_ratios)] = INF_REPLACEMENT
                            ot_performance_ratios[np.isinf(ot_performance_ratios)] = INF_REPLACEMENT
                            
                            # Calculate mean and SEM across all trials for line plot
                            mean_ratio = np.nanmean(performance_ratios)
                            sem_ratio = stats.sem(performance_ratios.flatten())

                            offtarget_mean_ratio = np.nanmean(ot_performance_ratios)
                            offtarget_sem_ratio = stats.sem(ot_performance_ratios.flatten())
                            
                            performance_ratios_mean.append(mean_ratio)
                            performance_ratios_sem.append(sem_ratio)

                            offtarget_performance_ratios_mean.append(offtarget_mean_ratio)
                            offtarget_performance_ratios_sem.append(offtarget_sem_ratio)
                            
                            # Store per-meta-trial means for AUC computation
                            # Shape: (n_meta,) - mean across trials within each meta trial
                            meta_performance_ratios.append(np.nanmean(performance_ratios, axis=1))
                            meta_offtarget_performance_ratios.append(np.nanmean(ot_performance_ratios, axis=1))
                        
                        if labels[j]=='Act':
                            color = 'blue'
                        else:
                            color = 'black'
                        
                        # Convert to arrays for easier handling
                        performance_ratios_mean = np.array(performance_ratios_mean)
                        performance_ratios_sem = np.array(performance_ratios_sem)
                        offtarget_performance_ratios_mean = np.array(offtarget_performance_ratios_mean)
                        offtarget_performance_ratios_sem = np.array(offtarget_performance_ratios_sem)
                        
                        # Check for NaN and inf values that would cause disconnected lines
                        has_nan_target = np.any(np.isnan(performance_ratios_mean))
                        has_nan_offtarget = np.any(np.isnan(offtarget_performance_ratios_mean))
                        has_inf_target = np.any(np.isinf(performance_ratios_mean))
                        has_inf_offtarget = np.any(np.isinf(offtarget_performance_ratios_mean))
                        if has_nan_target or has_nan_offtarget or has_inf_target or has_inf_offtarget:
                            print(f"Warning: NaN/inf values detected in layer {layer} for {labels[j]}")
                            print(f"  Target NaNs: {np.sum(np.isnan(performance_ratios_mean))}/{len(performance_ratios_mean)}, infs: {np.sum(np.isinf(performance_ratios_mean))}")
                            print(f"  Offtarget NaNs: {np.sum(np.isnan(offtarget_performance_ratios_mean))}/{len(offtarget_performance_ratios_mean)}, infs: {np.sum(np.isinf(offtarget_performance_ratios_mean))}")
                        
                        plt.plot(PCTS, performance_ratios_mean, label='Target', color=color)
                        plt.fill_between(PCTS, performance_ratios_mean - performance_ratios_sem, performance_ratios_mean + performance_ratios_sem, color=color, alpha=0.5)
                        plt.plot(PCTS, offtarget_performance_ratios_mean, label='Offtarget', color='r')
                        plt.fill_between(PCTS, offtarget_performance_ratios_mean - offtarget_performance_ratios_sem, offtarget_performance_ratios_mean + offtarget_performance_ratios_sem, color='r', alpha=0.5)
                        plt.tight_layout()
                        F.QT()
                        F.save(f'{FIG}_layer_{layer}_{labels[j]}.svg')
                        plt.close()
                        
                        # Compute AUC for each meta trial
                        # meta_performance_ratios has shape (n_pcts, n_meta)
                        # Transpose to (n_meta, n_pcts) for easier iteration
                        meta_performance_ratios = np.array(meta_performance_ratios).T  # Shape: (n_meta, n_pcts)
                        meta_offtarget_performance_ratios = np.array(meta_offtarget_performance_ratios).T
                        
                        # Compute AUC for each meta trial
                        targ_auc = np.array([bscope.compute_auc(PCTS, meta_performance_ratios[m, :]) 
                                            for m in range(N_META_TRIALS)])
                        offtarg_auc = np.array([bscope.compute_auc(PCTS, meta_offtarget_performance_ratios[m, :]) 
                                               for m in range(N_META_TRIALS)])

                        targs.append(targ_auc)  # Shape: (n_meta,)
                        offtargs.append(offtarg_auc)  # Shape: (n_meta,)

                        # Compute difference for each meta trial
                        d = (targ_auc - offtarg_auc) / offtarg_auc  # Shape: (n_meta,)
                        differences.append(d)
                    ddiffs.append(differences)  # Shape: (n_layers, n_meta)

                # ddiffs has shape (2, n_layers, n_meta)
                # Convert to arrays for easier manipulation
                ddiffs_cont = np.array(ddiffs[0])  # Shape: (n_layers, n_meta)
                ddiffs_act = np.array(ddiffs[1])  # Shape: (n_layers, n_meta)
                
                # Compute mean across meta trials for line plot
                ddiffs_cont_mean = np.nanmean(ddiffs_cont, axis=1)  # Shape: (n_layers,)
                ddiffs_act_mean = np.nanmean(ddiffs_act, axis=1)  # Shape: (n_layers,)
                
                # Check for NaN and inf values
                has_nan_cont = np.any(np.isnan(ddiffs_cont_mean))
                has_nan_act = np.any(np.isnan(ddiffs_act_mean))
                has_inf_cont = np.any(np.isinf(ddiffs_cont_mean))
                has_inf_act = np.any(np.isinf(ddiffs_act_mean))
                if has_nan_cont or has_nan_act or has_inf_cont or has_inf_act:
                    print(f"Warning: NaN/inf values in AUC difference for {model_type}_{layer_type}_{perturbation_type}_top{topk}")
                    print(f"  Cont NaNs: {np.sum(np.isnan(ddiffs_cont_mean))}/{len(ddiffs_cont_mean)}, infs: {np.sum(np.isinf(ddiffs_cont_mean))}")
                    print(f"  Act NaNs: {np.sum(np.isnan(ddiffs_act_mean))}/{len(ddiffs_act_mean)}, infs: {np.sum(np.isinf(ddiffs_act_mean))}")

                print ('ddiffs_cont_mean:', ddiffs_cont_mean)
                
                # Compute SEM across meta trials for error bands
                ddiffs_cont_sem = stats.sem(ddiffs_cont, axis=1, nan_policy='omit')  # Shape: (n_layers,)
                ddiffs_act_sem = stats.sem(ddiffs_act, axis=1, nan_policy='omit')  # Shape: (n_layers,)
                
                # Plot mean lines WITHOUT markers on the actual data
                line_cont, = plt.plot(LAYERS, ddiffs_cont_mean, color='k', linewidth=1.5)
                line_act, = plt.plot(LAYERS, ddiffs_act_mean, color='b', linewidth=1.5)
                
                # Add error bands (SEM)
                plt.fill_between(LAYERS, ddiffs_cont_mean - ddiffs_cont_sem, ddiffs_cont_mean + ddiffs_cont_sem, color='k', alpha=0.2)
                plt.fill_between(LAYERS, ddiffs_act_mean - ddiffs_act_sem, ddiffs_act_mean + ddiffs_act_sem, color='b', alpha=0.2)
                
                # Plot individual meta trial points (without labels so they don't clutter legend)
                for m in range(N_META_TRIALS):
                    plt.scatter(LAYERS, ddiffs_cont[:, m], marker='x', color='k', s=30, alpha=0.4)
                    plt.scatter(LAYERS, ddiffs_act[:, m], marker='o', color='b', s=20, alpha=0.4)
                
                # Create custom legend entries with markers
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='k', marker='x', markersize=8, linewidth=1.5, label='Cont'),
                    Line2D([0], [0], color='b', marker='o', markersize=6, linewidth=1.5, label='Act')
                ]
                
                n_layers = len(LAYERS)
                if n_layers <= 1:
                    ticks = [0]
                else:
                    ticks = np.linspace(0, n_layers - 1, num=5)
                    ticks = np.round(ticks).astype(int)
                    ticks = np.unique(ticks)
                plt.xticks(ticks, [str(t) for t in ticks])
                plt.axhline(0, color='gray', linestyle='--', linewidth=1)
                plt.ylabel('AUC(Target) - AUC(OffTarget)')
                plt.xlabel('Layer')
                plt.legend(handles=legend_elements)
                plt.tight_layout()
                F.QT()
                F.save(f'{FIG}_auc_diff_{TOPK}.svg')
                plt.close()



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
