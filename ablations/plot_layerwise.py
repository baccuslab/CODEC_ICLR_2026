import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from skkm import Figaro

F = Figaro('cont', extension='svg')
topk=1

if topk==1:
    TOPK = 'top1'
else:
    TOPK = 'top5'
# Single run
RESULTS_FILE = '/home/jbmelander/Misc/rn50_pos_cont.pkl'
with open(RESULTS_FILE, 'rb') as f:

    perturbation_data = pickle.load(f)

# Extract layers and percentages from the data
LAYERS = list(perturbation_data.keys())
PCTS = list(perturbation_data[LAYERS[0]].keys())

# Create a figure with subplots for each layer

for i, layer in enumerate(LAYERS):
    
    # Arrays to store means and SEMs
    performance_ratios_mean = []
    performance_ratios_sem = []

    offtarget_performance_ratios_mean = []
    offtarget_performance_ratios_sem = []
    
    for pct in PCTS:
        # Extract original and perturbed accuracies for this layer and percentage
        og_accs = np.array(perturbation_data[layer][pct]['og_{}'.format(TOPK)])
        pert_accs = np.array(perturbation_data[layer][pct]['pert_{}'.format(TOPK)])

        print(og_accs.shape)
        print(pert_accs.shape)

        input()
        
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
        mean_ratio = np.mean(performance_ratios)
        sem_ratio = stats.sem(performance_ratios)

        offtarget_mean_ratio = np.mean(ot_performance_ratios)
        offtarget_sem_ratio = stats.sem(ot_performance_ratios)
        
        performance_ratios_mean.append(mean_ratio)
        performance_ratios_sem.append(sem_ratio)

        offtarget_performance_ratios_mean.append(offtarget_mean_ratio)
        offtarget_performance_ratios_sem.append(offtarget_sem_ratio)
    
    # Plot with error bars
    plt.errorbar(PCTS, performance_ratios_mean, yerr=performance_ratios_sem, 
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=6)

    plt.errorbar(PCTS, offtarget_performance_ratios_mean, yerr=offtarget_performance_ratios_sem,
                marker='s', capsize=5, capthick=2, linewidth=2, markersize=6, color='orange', label='Off-target')

    ax = plt.gca()
    
    ax.set_xlabel('Percentage of Channels Kept (%)')
    ax.set_ylabel('Fraction of Target Class Performance')
    ax.set_title(f'Layer {layer} Ablation Results')
    ax.grid(True, alpha=0.3)
    
    # Add a horizontal line at y=1 for reference (no performance loss)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No Performance Loss')
    F.QT()
    F.save('layerwise_ablation_' + str(layer))
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
        og_accs = np.array(perturbation_data[layer][pct]['og_{}'.format(TOPK)])
        pert_accs = np.array(perturbation_data[layer][pct]['pert_{}'.format(TOPK)])
        
        target_og = og_accs[:, 0]
        target_pert = pert_accs[:, 0]
        performance_ratios = target_pert / target_og
        
        performance_ratios_mean.append(np.mean(performance_ratios))
        performance_ratios_sem.append(stats.sem(performance_ratios))
    plt.plot(PCTS, performance_ratios_mean)
    # plt.errorbar(PCTS, performance_ratios_mean, yerr=performance_ratios_sem, 
    #             marker='o', capsize=3, capthick=1, linewidth=2, markersize=4,
    #             color=colors[i], label=f'Layer {layer}')

plt.xlabel('Percentage of Channels Ablated(%)')
plt.ylabel('Fraction of Target Class Performance')
plt.title('All Layers: Target Class Performance vs Channels Kept')
plt.grid(True, alpha=0.3)
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No Performance Loss')
plt.legend()
plt.tight_layout()
F.QT()
F.save('layerwise_ablation_summary')

