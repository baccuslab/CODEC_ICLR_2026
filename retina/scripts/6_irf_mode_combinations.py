import os
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import torch
import torchdeepretina as tdr
from torch.utils.data import DataLoader, TensorDataset
import tqdm
from matplotlib import cm
import bscope
from itertools import combinations
from collections import Counter, defaultdict

def select_timepoints(examples, max_examples):
    """
    Select diverse examples to avoid clustering
    
    Args:
        examples: list of example dicts sorted by activation strength
        max_examples: number of examples to select
    
    Returns:
        selected_examples: list of diverse examples
    """
    if len(examples) <= max_examples:
        return examples
    
    # Start with the highest activation example
    selected = [examples[0]]
    remaining = examples[1:]
    
    # Greedily select examples that are furthest from already selected ones
    while len(selected) < max_examples and remaining:
        best_candidate = None
        max_min_distance = -1
        
        for candidate in remaining:
            # Find minimum distance to any already selected example
            min_distance = min(abs(candidate['timepoint'] - sel['timepoint']) 
                             for sel in selected)
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_candidate = candidate
        
        if best_candidate:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break
    
    return selected

def crop_around_peak_20x20(spatial_filter):
    """Crop spatial filter to 20x20 centered on the peak pixel"""
    # Find peak pixel location
    abs_filter = np.abs(spatial_filter)
    peak_row, peak_col = np.unravel_index(np.argmax(abs_filter), abs_filter.shape)
    
    # Calculate 20x20 crop bounds centered on peak
    crop_size = 20
    half_size = crop_size // 2
    
    # Get original dimensions
    h, w = spatial_filter.shape
    
    # Calculate crop bounds
    r1 = max(0, peak_row - half_size)
    r2 = min(h, peak_row + half_size)
    c1 = max(0, peak_col - half_size) 
    c2 = min(w, peak_col + half_size)
    
    # Ensure exactly 20x20 by padding if needed
    cropped = spatial_filter[r1:r2, c1:c2]
    
    # Pad to exactly 20x20 if crop hit boundaries
    pad_top = max(0, half_size - (peak_row - r1))
    pad_bottom = max(0, half_size - (r2 - peak_row))
    pad_left = max(0, half_size - (peak_col - c1))
    pad_right = max(0, half_size - (c2 - peak_col))
    
    cropped = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    
    return cropped

def list_available_cells(file_path):
    """List all cells available in the HDF5 file"""
    with h5.File(file_path, 'r') as f:
        cell_groups = [key for key in f.keys() if key.startswith('cell_')]
        cell_ids = [int(key.split('_')[1]) for key in cell_groups]
        return sorted(cell_ids)

def load_irfs(file_path, cell_id, load_ig_irf=False):
    """
    Load instantaneous receptive field data for a specific cell from the saved HDF5 file.

    Args:
        file_path: Path to the HDF5 file
        cell_id: ID of the cell to load
        load_ig_irf: if True, also load ig_irf_mean and ig_irf_all (default False);
                     set to False to load only regular IRF data

    Returns:
        dict with 'regular_irf_mean', 'regular_irf_all', and optionally 'ig_irf_mean', 'ig_irf_all'
    """
    with h5.File(file_path, 'r') as f:
        cell_grp = f[f'cell_{cell_id}']

        result = {
            'regular_irf_mean': cell_grp['regular_irf_mean'][:],
            'regular_irf_all': cell_grp['regular_irf_all'][:],
            'cell_id': cell_grp.attrs['cell_id'],
            'cell_index': cell_grp.attrs['cell_index']
        }

        # Only load IG-IRF if requested and it exists
        if load_ig_irf and 'ig_irf_mean' in cell_grp:
            result['ig_irf_mean'] = cell_grp['ig_irf_mean'][:]
            result['ig_irf_all'] = cell_grp['ig_irf_all'][:]

        return result

def find_mode_combinations(loadings, min_modes=2, max_modes=5, min_activation_threshold=0.001):
    """
    Find all unique mode combinations that occur together
    
    Args:
        loadings: (n_timepoints, n_atoms) array of mode activations
        min_modes: minimum number of modes that must be active
        max_modes: maximum number of modes to consider
        min_activation_threshold: minimum activation level to count as "active"
    
    Returns:
        combo_counter: Counter of mode combinations and their frequencies
        combo_timepoints: dict mapping combinations to lists of timepoints
    """
    combo_counter = Counter()
    combo_timepoints = defaultdict(list)
    
    for t in range(loadings.shape[0]):
        active_modes = np.where(loadings[t, :] > min_activation_threshold)[0]
        
        if min_modes <= len(active_modes) <= max_modes:
            # Convert to tuple for hashing
            combo = tuple(sorted(active_modes))
            combo_counter[combo] += 1
            combo_timepoints[combo].append(t)
    
    return combo_counter, combo_timepoints

def analyze_mode_combination_irfs(combo, combo_timepoints, all_irf_data, preds, test_data, 
                                 loadings, min_cell_response=1, max_examples_per_cell=6):
    """
    Analyze IRFs for a specific mode combination across all cells
    
    Args:
        combo: tuple of mode indices (e.g., (3, 7, 12))
        combo_timepoints: list of timepoints when this combo occurs
        all_irf_data: dictionary of IRF data for all cells
        preds: model predictions
        test_data: test dataset
        loadings: mode activation loadings
        min_cell_response: minimum cell response to include
        max_examples_per_cell: max examples to show per cell
    """
    combo_str = f"Modes {combo}"
    print(f"\nAnalyzing {combo_str} ({len(combo_timepoints)} total occurrences)")
    
    # Find cells that respond to this combination
    responding_cells = {}
    
    for cell_id in all_irf_data.keys():
        if cell_id not in test_data.cells:
            continue
            
        cell_idx = test_data.cells.index(cell_id)
        cell_responses = preds[:, cell_idx]
        
        # Find timepoints where this combo occurs AND cell is responding
        valid_examples = []
        
        for t in combo_timepoints:
            if cell_responses[t] > min_cell_response:
                # Get activation strengths for the modes in this combo
                mode_activations = loadings[t, list(combo)]
                total_activation = np.sum(mode_activations)
                
                valid_examples.append({
                    'timepoint': t,
                    'mode_activations': mode_activations,
                    'total_activation': total_activation,
                    'cell_response': cell_responses[t]
                })
        
        if len(valid_examples) > 0:
            # Sort by total activation strength first
            valid_examples_sorted = sorted(valid_examples, 
                                        key=lambda x: x['total_activation'], reverse=True)
            
            # Then select diverse timepoints if needed
            if len(valid_examples_sorted) > max_examples_per_cell:
                selected_examples = select_timepoints(valid_examples_sorted, max_examples_per_cell)
            else:
                selected_examples = valid_examples_sorted
                
            responding_cells[cell_id] = selected_examples
    
    if len(responding_cells) == 0:
        print(f"No cells respond strongly to {combo_str}")
        return
    
    print(f"Found {len(responding_cells)} cells responding to {combo_str}")
    
    # Create visualization
    max_examples_per_cell = min(max_examples_per_cell, 
                               max(len(examples) for examples in responding_cells.values()))
    
    cells_with_examples = list(responding_cells.keys())
    n_cells = len(cells_with_examples)
    
    fig, axes = plt.subplots(n_cells, max_examples_per_cell, 
                            figsize=(3*max_examples_per_cell, 3*n_cells))
    
    # Handle single cell or single example cases
    if n_cells == 1 and max_examples_per_cell == 1:
        axes = np.array([[axes]])
    elif n_cells == 1:
        axes = axes.reshape(1, -1)
    elif max_examples_per_cell == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'{combo_str}: IRFs when modes co-activate', fontsize=16)
    
    for row_idx, cell_id in enumerate(cells_with_examples):
        examples = responding_cells[cell_id]
        
        # Row label
        axes[row_idx, 0].text(-0.15, 0.5, 
                             f'Cell {cell_id}\n({len(examples)} examples)', 
                             transform=axes[row_idx, 0].transAxes, 
                             rotation=90, va='center', ha='right', fontsize=10)
        
        for col_idx in range(max_examples_per_cell):
            ax = axes[row_idx, col_idx] if len(axes.shape) == 2 else axes[col_idx]
            
            if col_idx < len(examples):
                example = examples[col_idx]
                timepoint = example['timepoint']
                mode_activations = example['mode_activations']
                total_activation = example['total_activation']
                cell_response = example['cell_response']
                
                # Get IRF for this cell at this timepoint
                regular_irf_all = all_irf_data[cell_id]['regular_irf_all']
                irf_at_time = regular_irf_all[timepoint]
                
                # Decompose to spatial component
                decomp = bscope.decompose(irf_at_time, k=1)
                spatial_filter = decomp[0][0]
                
                # Crop around peak
                cropped_filter = crop_around_peak_20x20(spatial_filter)
                
                # Plot
                clim = np.max(np.abs(cropped_filter))
                if clim > 0:
                    im = ax.imshow(cropped_filter, cmap='RdBu_r', vmin=-clim, vmax=clim)
                
                # Create title with mode activations
                mode_str = ', '.join([f'{combo[i]}:{mode_activations[i]:.2f}' 
                                    for i in range(len(combo))])
                ax.set_title(f't={timepoint}\n[{mode_str}]\nfiring rate ={cell_response:.1f} hz', 
                           fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                
            else:
                # Empty subplot
                ax.axis('off')
    
    plt.tight_layout()
    # output_dir=f'/home/zalaoui/mode_combination_irfs_{identifier}_{layer_name}'
    # os.makedirs(output_dir, exist_ok=True)
    # plt.savefig(os.path.join(output_dir, f'mode_combination_{"_".join(map(str, combo))}_irfs.svg'))
    plt.show()



# =============================================================================
# MAIN ANALYSIS LOOP
# =============================================================================

identifier = '15-10-07_naturalscene'
model_path = f"/home/zalaoui/torch-deep-retina/models/{identifier}.pt"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load retinal model
checkpt = tdr.io.load_checkpoint(model_path)
model = tdr.io.load_model(model_path)
model = tdr.utils.stacked2conv(model)
model.to(device)
model.eval()

# Extract model parameters
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

# Compute model responses
bsize = 500
model_response = tdr.utils.inspect(model, test_data.X, batch_size=bsize, to_numpy=True)
preds = model_response['outputs']
truth = test_data.y
print("Preds shape:", preds.shape)
print("Truth shape:", truth.shape)

# Load SAE results
singletarget = 'surprisal'
saveflag = 'noL!' if 'naturalscene' in identifier else 'josh'
layer_name = 'conv0'
sae_results_path = f'/home/zalaoui/retinal_codec/{identifier}_codec/single_target_saes/{singletarget}_sae_results_{saveflag}.h5'

with h5.File(sae_results_path, 'r') as f:
    loadings = f[layer_name]['codes'][:]
    dictionary = f[layer_name]['dictionary'][:]
    original_attributions = f[layer_name]['original_attributions'][:]
    scalar_target_sums = f['target_data/scalar_target_sums'][:]
    code_activity = f[layer_name]['code_activity'][:]
    r2_score = f[layer_name].attrs['r2_score']
    n_features = f[layer_name].attrs['n_features']

n_timepoints, n_modes = loadings.shape


print(f"Loadings shape: {loadings.shape}, Dictionary shape: {dictionary.shape}")

# Load all IRF data once
file_path = f'/home/zalaoui/retina_codec/{identifier}/irf_results/{identifier}_irfs.h5'
available_cells = list_available_cells(file_path)
print(f"Available cells: {available_cells}")

print("Loading all IRF data...")
all_irf_data = {cell_id: load_irfs(file_path, cell_id) for cell_id in available_cells}
print("All IRF data loaded!")


print("Finding Only 1 Mode Active...")
single_counter, single_timepoints = find_mode_combinations(
    loadings, 
    min_modes=1,      # At least 2 modes active
    max_modes=1,      # At most 5 modes active
    min_activation_threshold=0.01
)



# Sort combinations by frequency
most_common_single_mode = single_counter.most_common(1)  # Top 20 most frequent combinations


print(f"\nMost common mode:")
print("="*50)
for mode_idx, count in most_common_single_mode:
    print(f"Mode {mode_idx}: {count} occurrences")



for combo, count in most_common_single_mode:
    timepoints = single_timepoints[combo]
    analyze_mode_combination_irfs(
        combo=combo,
        combo_timepoints=timepoints,
        all_irf_data=all_irf_data,
        preds=preds,
        test_data=test_data,
        loadings=loadings,
        min_cell_response=1,
        max_examples_per_cell=6
    )

    print("\n" + "-"*60 + "\n")

# --- Multi-mode analysis (1–3 modes active) ---
print("Finding 1-3 Modes Active...")
multi_counter, multi_timepoints = find_mode_combinations(
    loadings,
    min_modes=1,
    max_modes=4,
    min_activation_threshold=0.01
)

print(f"Found {len(multi_counter)} unique mode combinations.")
most_common_multi_mode = multi_counter.most_common(20)

print(f"\nMost common 1-3 mode combinations:")
print("="*50)
for combo, count in most_common_multi_mode[:10]:
    print(f"Modes {combo}: {count} occurrences")

print("\n" + "="*80)
print("ANALYZING TOP 1-3 MODE COMBINATIONS")
print("="*80)

for combo, count in most_common_multi_mode:
    timepoints = multi_timepoints[combo]
    analyze_mode_combination_irfs(
        combo=combo,
        combo_timepoints=timepoints,
        all_irf_data=all_irf_data,
        preds=preds,
        test_data=test_data,
        loadings=loadings,
        min_cell_response=1,
        max_examples_per_cell=6
    )

    print("\n" + "-"*60 + "\n")

