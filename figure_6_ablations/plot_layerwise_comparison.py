
from IPython import embed
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import skkm

figaro = skkm.Figaro('figure_6b', extension='svg')
# Single run
ACT_RESULTS_FILE = '/home/jbmelander/preserve_rn50_pos_act.pkl'
CON_RESULTS_FILE = '/home/jbmelander/preserve_rn50_pos_cont.pkl'

TOPK = 'top5'


with open(ACT_RESULTS_FILE, 'rb') as f:
    act_perturbation_data = pickle.load(f)
with open(CON_RESULTS_FILE, 'rb') as f:
    con_perturbation_data = pickle.load(f)

# Extract layers and percentages from the data
LAYERS = list(con_perturbation_data.keys())
PCTS = list(con_perturbation_data[LAYERS[0]].keys())

# Create a figure with subplots for each layer
for i, layer in enumerate(LAYERS):
    
    layer_data = {}
    for perturbation_data, title in zip([act_perturbation_data, con_perturbation_data], ['Activations', 'Contributions']):
        layer_data[title] = []
        # Arrays to store means and SEMs
        performance_ratios_mean = []
        performance_ratios_sem = []

        offtarget_performance_ratios_mean = []
        offtarget_performance_ratios_sem = []

        selectivity_mean = []
        selectivity_sem = []
        

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

            # # Calculate delta performance
            performance_ratios = target_pert / target_og
            ot_performance_ratios = offtarget_pert  / offtarget_og
            # performance_ratios = (target_og - target_pert) / target_og
            # ot_performance_ratios = (offtarget_og - offtarget_pert) / offtarget_og
            # performance_ratios = (target_pert/target_og)
            # ot_performance_ratios = (offtarget_pert / offtarget_og)


            # Calculate fraction of target class performance
            # performance_ratios = target_pert / target_og
            # ot_performance_ratios = offtarget_pert / offtarget_og

            # Calculate selectivity of the fraction performance
            selectivity = (performance_ratios - ot_performance_ratios) / (performance_ratios + ot_performance_ratios)
            selectivity_mean.append(np.nanmean(selectivity))
            selectivity_sem.append(stats.sem(selectivity, nan_policy='omit'))


            
            # Calculate mean and SEM
            mean_ratio = np.nanmean(performance_ratios)
            sem_ratio = stats.sem(performance_ratios, nan_policy='omit')

            offtarget_mean_ratio = np.nanmean(ot_performance_ratios)
            offtarget_sem_ratio = stats.sem(ot_performance_ratios, nan_policy='omit')
            
            performance_ratios_mean.append(mean_ratio)
            performance_ratios_sem.append(sem_ratio)

            offtarget_performance_ratios_mean.append(offtarget_mean_ratio)
            offtarget_performance_ratios_sem.append(offtarget_sem_ratio)

        layer_data[title] = (performance_ratios_mean, performance_ratios_sem,
                             offtarget_performance_ratios_mean, offtarget_performance_ratios_sem, selectivity_mean, selectivity_sem)


    fig, ax = plt.subplots(figsize=(8, 6))     
    for title in layer_data:
        color = 'blue' if title == 'Activations' else 'black'
        performance_ratios_mean, performance_ratios_sem, offtarget_performance_ratios_mean, offtarget_performance_ratios_sem, selectivity_mean, selectivity_sem = layer_data[title]
        print(len(performance_ratios_mean))
        ax.errorbar(PCTS, performance_ratios_mean, yerr=performance_ratios_sem, color=color,
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=6, label=f'Target - {title}')
        ax.errorbar(PCTS, offtarget_performance_ratios_mean, yerr=offtarget_performance_ratios_sem, color=color,
                    marker='s', capsize=5, capthick=2, linewidth=2, markersize=6, label=f'Off-target - {title}', linestyle='--')
    ax.set_xlabel('Percentage of Channels Ablated (%)')
    ax.set_ylabel('Fraction of Class Performance')
    ax.set_title(f'Layer {layer} Ablation Results')
    ax.grid(True, alpha=0.3)
    # ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No Performance Loss')
    figaro.save(f'layer_{layer}_ablation_comparison')

    for title in layer_data:
        color = 'blue' if title == 'Activations' else 'black'
        performance_ratios_mean, performance_ratios_sem, offtarget_performance_ratios_mean, offtarget_performance_ratios_sem, selectivity_mean, selectivity_sem = layer_data[title]
        plt.errorbar(PCTS, selectivity_mean, yerr=selectivity_sem, color=color,
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=6, label=f'Selectivity - {title}')
    plt.xlabel('Percentage of Channels Ablated (%)')
    plt.ylabel('Selectivity')
    plt.title(f'Layer {layer} Ablation Selectivity')
    figaro.save(f'layer_{layer}_ablation_selectivity_comparison')


        # # Plot with error bars
        # ax.errorbar(PCTS, performance_ratios_mean, yerr=performance_ratios_sem, 
        #             marker='o', capsize=5, capthick=2, linewidth=2, markersize=6)

        # ax.errorbar(PCTS, offtarget_performance_ratios_mean, yerr=offtarget_performance_ratios_sem,
        #             marker='s', capsize=5, capthick=2, linewidth=2, markersize=6, color='orange', label='Off-target')
        
        # ax.set_xlabel('Percentage of Channels Kept (%)')
        # ax.set_ylabel('Fraction of Target Class Performance')
        # ax.set_title(f'Layer {layer} Ablation Results')
        # ax.grid(True, alpha=0.3)
        # ax.set_ylim(0, 1.1)  # Performance ratio shouldn't exceed 1
        
        # # Add a horizontal line at y=1 for reference (no performance loss)
        # ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No Performance Loss')
        # ax.legend()

# Remove the extra subplot if there are fewer layers than subplots

# Print summary statistics
print("Summary Statistics:")
print("=" * 50)
for layer in LAYERS:
    print(f"\nLayer {layer}:")
    for pct in PCTS:
        og_accs = np.array(perturbation_data[layer][pct]['og_{}'.format(TOPK)])
        pert_accs = np.array(perturbation_data[layer][pct]['pert_{}'.format(TOPK)])
        
        target_og = og_accs[:, 0]
        target_pert = pert_accs[:, 0]
        performance_ratios = target_pert / target_og
        
        mean_ratio = np.mean(performance_ratios)
        sem_ratio = stats.sem(performance_ratios)
        
        print(f"  {pct}% channels: {mean_ratio:.3f} ± {sem_ratio:.3f}")

# Optional: Create a summary plot showing all layers together
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(LAYERS)))

for i, layer in enumerate(LAYERS):
    performance_ratios_mean = []
    performance_ratios_sem = []
    
    for pct in PCTS:
        og_accs = np.array(act_perturbation_data[layer][pct]['og_{}'.format(TOPK)])
        pert_accs = np.array(act_perturbation_data[layer][pct]['pert_{}'.format(TOPK)])
        
        target_og = og_accs[:, 0]
        target_pert = pert_accs[:, 0]
        performance_ratios = target_pert / target_og
        
        performance_ratios_mean.append(np.mean(performance_ratios))
        performance_ratios_sem.append(stats.sem(performance_ratios))
    
    plt.errorbar(PCTS, performance_ratios_mean, yerr=performance_ratios_sem, 
                marker='o', capsize=3, capthick=1, linewidth=2, markersize=4,
                color=colors[i], label=f'Layer {layer}')

plt.xlabel('Percentage of Channels Ablated(%)')
plt.ylabel('Fraction of Target Class Performance')
plt.title('All Layers: Target Class Performance vs Channels Kept')
plt.grid(True, alpha=0.3)
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No Performance Loss')
plt.legend()
plt.tight_layout()
figaro.save('act_all_layers_ablation_comparison')

for i, layer in enumerate(LAYERS):
    performance_ratios_mean = []
    performance_ratios_sem = []
    
    for pct in PCTS:
        og_accs = np.array(con_perturbation_data[layer][pct]['og_{}'.format(TOPK)])
        pert_accs = np.array(con_perturbation_data[layer][pct]['pert_{}'.format(TOPK)])
        
        target_og = og_accs[:, 0]
        target_pert = pert_accs[:, 0]
        performance_ratios = target_pert / target_og
        
        performance_ratios_mean.append(np.mean(performance_ratios))
        performance_ratios_sem.append(stats.sem(performance_ratios))
    
    plt.errorbar(PCTS, performance_ratios_mean, yerr=performance_ratios_sem, 
                marker='o', capsize=3, capthick=1, linewidth=2, markersize=4,
                color=colors[i], label=f'Layer {layer}')

plt.xlabel('Percentage of Channels Ablated(%)')
plt.ylabel('Fraction of Target Class Performance')
plt.title('All Layers: Target Class Performance vs Channels Kept')
plt.grid(True, alpha=0.3)
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No Performance Loss')
plt.legend()
plt.tight_layout()
figaro.save('con_all_layers_ablation_comparison')

