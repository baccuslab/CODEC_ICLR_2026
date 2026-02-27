import os
import numpy as np
import torch
import torchdeepretina as tdr
import matplotlib.pyplot as plt
import h5py as h5
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# =============================================================================
# LOAD MODEL AND DATA
# =============================================================================
identifier = '15-11-21b_naturalscene'
model_path = f"/home/zalaoui/torch-deep-retina/models/{identifier}.pt"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

checkpt = tdr.io.load_checkpoint(model_path)
model = tdr.io.load_model(model_path)
model = tdr.utils.stacked2conv(model)
model.to(device)
model.eval()

dataset = checkpt['dataset'] 
cells = checkpt['cells']
stim_type = checkpt["stim_type"]
window_size = checkpt['img_shape'][0]
height = checkpt['img_shape'][1]
width = checkpt['img_shape'][2]
path_to_data = "/data/retina"
norm_stats = [checkpt['norm_stats']['mean'], checkpt['norm_stats']['std']]

test_data = tdr.datas.loadexpt(dataset, cells, stim_type, 'test',
                               window_size, nskip=0,
                               norm_stats=norm_stats,
                               data_path=path_to_data)

bsize = 500
model_response = tdr.utils.inspect(model, test_data.X, batch_size=bsize, to_numpy=True)
preds = model_response['outputs']
truth = test_data.y
pearsons = tdr.utils.pearsonr(preds, truth)

# =============================================================================
# LOAD CODES FOR BOTH LAYERS
# =============================================================================
singletarget = 'surprisal'
if 'whitenoise' in identifier:
    saveflag = 'josh'
elif '15-10-07' in identifier:
    saveflag = 'noL!'
elif '15-11-21b' in identifier:
    saveflag = 'steve'

layer_name0 = 'conv0'
layer_name1 = 'conv1'
sae_results_path = f'/home/zalaoui/retinal_codec/{identifier}_codec/single_target_saes/{singletarget}_sae_results_{saveflag}.h5'

with h5.File(sae_results_path, 'r') as f:
    codes_layer0 = f[layer_name0]['codes'][:]
    print(f"Loaded codes for {layer_name0} with shape {codes_layer0.shape}")
    codes_layer1 = f[layer_name1]['codes'][:]
    print(f"Loaded codes for {layer_name1} with shape {codes_layer1.shape}")



# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================
def analyze_layer(codes, layer_name, preds, test_data):
    _, n_atoms = codes.shape
    n_cells = len(test_data.cells)
    mode_response_matrix = np.full((n_cells, n_atoms), np.nan)
    
    for cell_idx, cell_id in enumerate(test_data.cells):
        pred_responses_cell = preds[:, cell_idx]
        for atom_idx in range(n_atoms):
            atom_loading = codes[:, atom_idx]
            correlation = np.corrcoef(atom_loading, pred_responses_cell)[0, 1]
            mode_response_matrix[cell_idx, atom_idx] = correlation
    
    mode_matrix_filled = np.nan_to_num(mode_response_matrix, nan=0.0)
    scaler = StandardScaler()
    mode_matrix_scaled = scaler.fit_transform(mode_matrix_filled)
    
    return {
        'layer_name': layer_name,
        'mode_matrix_scaled': mode_matrix_scaled,
        'mode_matrix_filled': mode_matrix_filled
    }

# =============================================================================
# ANALYZE BOTH LAYERS
# =============================================================================
results_conv0 = analyze_layer(codes_layer0, 'conv0', preds, test_data)
results_conv1 = analyze_layer(codes_layer1, 'conv1', preds, test_data)

# =============================================================================
# K-MEANS CLUSTERING AND PLOTTING
# =============================================================================

from sklearn.metrics import silhouette_score

def find_optimal_k(data, k_min=2, k_max=8, random_state=42):
    """Find optimal k for k-means using silhouette score."""
    best_k = k_min
    best_score = -1
    scores = []
    for k in range(k_min, k_max+1):
        
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(data)
        if len(np.unique(labels)) == 1:
            score = -1
        else:
            score = silhouette_score(data, labels)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, scores

# Find optimal k for each layer
optimal_k_conv0, sil_scores_conv0 = find_optimal_k(results_conv0['mode_matrix_scaled'])
optimal_k_conv1, sil_scores_conv1 = find_optimal_k(results_conv1['mode_matrix_scaled'])

print(f"Optimal k for conv0: {optimal_k_conv0} (silhouette scores: {sil_scores_conv0})")
print(f"Optimal k for conv1: {optimal_k_conv1} (silhouette scores: {sil_scores_conv1})")




# Do k-means clustering for each layer
kmeans_conv0 = KMeans(n_clusters=optimal_k_conv0, random_state=42, n_init=10)
conv0_labels = kmeans_conv0.fit_predict(results_conv0['mode_matrix_scaled'])

kmeans_conv1 = KMeans(n_clusters=optimal_k_conv1, random_state=42, n_init=10)
conv1_labels = kmeans_conv1.fit_predict(results_conv1['mode_matrix_scaled'])

# Print cluster assignments
print("CONV0 K-means clusters:")
for i, cell_id in enumerate(test_data.cells):
    print(f"Cell {cell_id}: Cluster {conv0_labels[i]}")

print("\nCONV1 K-means clusters:")
for i, cell_id in enumerate(test_data.cells):
    print(f"Cell {cell_id}: Cluster {conv1_labels[i]}")

# Get conv0's row ordering and shared cell IDs
conv0_sort_indices = np.argsort(conv0_labels)
shared_ordered_cell_ids = [test_data.cells[i] for i in conv0_sort_indices]  
cluster_colors = ['lightblue', 'lightgreen', 'lightcoral']

fig, axes = plt.subplots(1, 2, figsize=(24, 8))

def assign_colors(cluster_labels):
    """Assign colors: largest=red(2), medium=green(1), smallest=blue(0)"""
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    size_order = np.argsort(counts)[::-1]  # Reverse for largest first
    
    # Create mapping: largest cluster -> 2 (red), medium -> 1 (green), smallest -> 0 (blue)
    label_mapping = {}
    for new_label, old_label_idx in enumerate(size_order):
        old_label = unique_labels[old_label_idx]
        label_mapping[old_label] = new_label
    
    return np.array([label_mapping[label] for label in cluster_labels])


conv0_labels_colored = assign_colors(conv0_labels)
conv1_labels_colored = assign_colors(conv1_labels)
cluster_colors = ['lightblue', 'lightgreen', 'lightcoral']  # 0=blue, 1=green, 2=red



for plot_idx, (results, labels, title) in enumerate([
    (results_conv0, conv0_labels_colored, 'CONV0'),
    (results_conv1, conv1_labels_colored, 'CONV1')
]):

    matrix_ordered = results['mode_matrix_filled'][conv0_sort_indices, :]
    labels_ordered = labels[conv0_sort_indices]
    
    # Plot heatmap
    im = axes[plot_idx].imshow(matrix_ordered, aspect='auto', cmap='Reds', 
                              vmin=0, vmax=np.percentile(results['mode_matrix_filled'], 95))
    
    # Add colored markers on the left side of each row
    for i, (cell_id, cluster_id) in enumerate(zip(shared_ordered_cell_ids, labels_ordered)):
        axes[plot_idx].scatter(-0.8, i, 
                              c=cluster_colors[cluster_id], 
                              s=150,
                              marker='s',
                              alpha=0.9,
                              edgecolor='black',
                              linewidth=1)
    
    axes[plot_idx].set_xlabel('SAE Atoms')
    axes[plot_idx].set_ylabel('Ganglion Cells')
    axes[plot_idx].set_title(f'{title}: Cluster-Colored Rows (k=3)')
    axes[plot_idx].set_yticks(range(len(shared_ordered_cell_ids)))
    axes[plot_idx].set_yticklabels([f'Cell_{cid}' for cid in shared_ordered_cell_ids], fontsize=8)
    
    # Adjust x-limits to show the colored markers
    axes[plot_idx].set_xlim(-1.2, matrix_ordered.shape[1] - 0.5)
plt.colorbar(im, ax=axes[1], label='Correlation with SAE Atoms')
plt.tight_layout()
plt.show()

