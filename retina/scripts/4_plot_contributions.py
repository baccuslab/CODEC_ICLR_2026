import os
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import bopt
import os
import numpy as np
import torch
import torchdeepretina as tdr
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py as h5
import bopt
from voltron import revcorr
import bscope

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

def load_irfs(file_path, cell_id):
    """
    Load IRF data for a specific cell from the saved HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
        cell_id: ID of the cell to load
        
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
        
        # Only load IG-IRF if it exists
        if 'ig_irf_mean' in cell_grp:
            result['ig_irf_mean'] = cell_grp['ig_irf_mean'][:]
            result['ig_irf_all'] = cell_grp['ig_irf_all'][:]
        
        return result


def list_available_cells(file_path):
    """List all cells available in the HDF5 file"""
    with h5.File(file_path, 'r') as f:
        # Just return the cell IDs from the groups
        cell_groups = [key for key in f.keys() if key.startswith('cell_')]
        cell_ids = [int(key.split('_')[1]) for key in cell_groups]
        return sorted(cell_ids)




# LOAD MODEL AND DATA
# =============================================================================
# Model and data paths
identifier = '15-10-07_naturalscene'
model_path = f"/home/zalaoui/torch-deep-retina/models/{identifier}.pt"
data_path = "/data/retina"
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


test_data = tdr.datas.loadexpt(dataset, cells, stim_type, 'test', window_size, nskip=0,
                                            norm_stats=norm_stats,
                                            data_path=path_to_data)



bsize = 500
model_response = tdr.utils.inspect(model, test_data.X,
                                        batch_size=bsize,
                                        to_numpy=True)
preds = model_response['outputs'] 
print("Preds shape:", preds.shape)
truth = test_data.y
print("Truth shape:", truth.shape)
pearsons = tdr.utils.pearsonr(preds,truth)

# Retinal model parameters
singletarget = 'surprisal'
if 'whitenoise' in identifier:
    saveflag='josh'
elif 'naturalscene' in identifier:
    saveflag='noL!'
layer_name = 'conv1' 
sae_results_path = f'/home/zalaoui/retinal_codec/{identifier}_codec/single_target_saes/{singletarget}_sae_results_{saveflag}.h5'
with h5.File(sae_results_path, 'r') as f:
    codes = f[layer_name]['codes'][:]  # (n_timepoints, n_alive_atoms)
    dictionary = f[layer_name]['dictionary'][:]  # (n_alive_atoms, n_features)
    original_attributions = f[layer_name]['original_attributions'][:]
    scalar_target_sums = f['target_data/scalar_target_sums'][:]
    code_activity = f[layer_name]['code_activity'][:]
    r2_score = f[layer_name].attrs['r2_score']
    n_features = f[layer_name].attrs['n_features']



output_h5_path = f'/home/zalaoui/retinal_codec/{identifier}_codec/{identifier}_{singletarget}.h5'

with h5.File(output_h5_path, 'r') as f:
    layer_names = [name.decode('utf-8') for name in f['metadata/layer_names'][:]]
    contributions_by_layer = {}
    for layer_name in layer_names:
        contributions_by_layer[layer_name] = f[f'{singletarget}_contributions/{layer_name}'][:]
        print(f"Loaded {layer_name}: {contributions_by_layer[layer_name].shape}")


print("Loading single target contributions data...")


n_timepoints, n_atoms = codes.shape
selected_unit_id = 4
selected_cell_idx = test_data.cells.index(selected_unit_id)
model_responses_cell = preds[:,selected_cell_idx]
actual_responses_cell = truth[:,selected_cell_idx]


# Find one atom active at each timepoint but not the other
timepoint1, timepoint2 = 4563, 4657
timepoint_list = [timepoint1, timepoint2]
threshold = 0.1
window_size = 5 # Check ±5 samples around each timepoint

# Find atoms with differential activity
atom_tp1_only = None  # Active at tp1, not tp2
atom_tp2_only = None  # Active at tp2, not tp1

for atom_idx in range(n_atoms):

    tp1_window = codes[timepoint1-window_size:timepoint1+window_size+1, atom_idx]
    tp2_window = codes[timepoint2-window_size:timepoint2+window_size+1, atom_idx]
    
    tp1_active = np.any(tp1_window > threshold)  # Active anywhere in tp1 window
    tp2_active = np.any(tp2_window > threshold)  # Active anywhere in tp2 window
    
    if tp1_active and not tp2_active and atom_tp1_only is None:
        atom_tp1_only = atom_idx
    elif tp2_active and not tp1_active and atom_tp2_only is None:
        atom_tp2_only = atom_idx
    

    if atom_tp1_only is not None and atom_tp2_only is not None:
        break



# Plot firing rate and atoms with contribution heatmaps (first 2 layers only)
fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

# Define the plotting window
plot_start = 4500
plot_end = 4750

# Firing rate plot
axes[0].plot(np.arange(plot_start, plot_end), model_responses_cell[plot_start:plot_end], color='red', linewidth=2)
axes[0].set_ylabel('Firing Rate')
axes[0].set_title(f'Cell {selected_unit_id} Model Response')

# Mark the timepoints (adjust for window offset)
if plot_start <= timepoint1 < plot_end:
    axes[0].axvline(timepoint1, color='black', linestyle='--', alpha=0.5)
if plot_start <= timepoint2 < plot_end:
    axes[0].axvline(timepoint2, color='black', linestyle='--', alpha=0.5)

# Attribution heatmaps for first 2 layers only
for layer_idx, layer_name in enumerate(layer_names[:2]):
    ax = axes[layer_idx + 1]  
    
    # Get contributions for this layer: [timepoints, spatial_dims]
    layer_contribs = contributions_by_layer[layer_name]
    
    # Apply the plotting window
    layer_contribs_windowed = layer_contribs[plot_start:plot_end]
    
    max_abs_val = np.max(np.abs(layer_contribs_windowed))
    
    im = ax.imshow(layer_contribs_windowed.T, aspect='auto', cmap='PiYG', interpolation='none',
                vmin=-max_abs_val, vmax=max_abs_val, extent=[plot_start, plot_end-1, 0, layer_contribs_windowed.shape[1]])
    
    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.35)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Contribution', labelpad=8)
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    
    ax.set_title(f'{layer_name} - Attribution Heatmap')
    ax.set_ylabel('Channels') 

axes[3].plot(np.arange(plot_start, plot_end), codes[plot_start:plot_end, atom_tp1_only], color='blue', linewidth=2, 
             label=f'Atom {atom_tp1_only} (active at tp1)')


axes[3].plot(np.arange(plot_start, plot_end), codes[plot_start:plot_end, atom_tp2_only], color='orange', linewidth=2, 
             label=f'Atom {atom_tp2_only} (active at tp2)')

# Mark the timepoints on the atom plot as well
if plot_start <= timepoint1 < plot_end:
    axes[3].axvline(timepoint1, color='black', linestyle='--', alpha=0.5)
if plot_start <= timepoint2 < plot_end:
    axes[3].axvline(timepoint2, color='black', linestyle='--', alpha=0.5)

axes[3].set_ylabel('Atom Loadings')
axes[3].legend() 

# Set x-label on the last subplot (atoms)
axes[3].set_xlabel('Time')

plt.tight_layout()
plt.show()


file_path = f'/home/zalaoui/retina_codec/{identifier}/irf_results/{identifier}_irfs.h5'

print(f"Loading IRF data from: {file_path}")

# Check what cells are available
available_cells = list_available_cells(file_path)


# Plot individual cell decompositions for first 3 cells
for cell_id in available_cells:

    if cell_id != selected_unit_id:
        continue

    # Load data for this cell
    data = load_irfs(file_path, cell_id)
    print(f"Cell ID: {data['cell_id']}, Cell Index: {data['cell_index']}")
    ig_irf_all = data['ig_irf_all']
    regular_irf_all = data['regular_irf_all']
    print(f"IG IRF All Shape: {ig_irf_all.shape}, Regular IRF All Shape: {regular_irf_all.shape}")



for cell_id in available_cells:

    if cell_id != selected_unit_id:
        continue

    # Load data for this cell
    data = load_irfs(file_path, cell_id)
    print(f"Cell ID: {data['cell_id']}, Cell Index: {data['cell_index']}")
    regular_irf_all = data['regular_irf_all']
    print(f"IRF All Shape: {regular_irf_all.shape}")

    for timepoint in timepoint_list:

        print(f"Plotting IRFs for at timepoint {timepoint}")
        

        regular_irf_atom_at_time = regular_irf_all[timepoint]  # was: pure_timepoint
        
        fig, ax = plt.subplots(1,2)

        fig.suptitle(f'spatial component, IRF at {timepoint} for cell {cell_id}')
        decomp = bopt.decompose(regular_irf_atom_at_time, k=1)
        cropped_decomp=crop_around_peak_20x20(decomp[0][0])
        clim = np.max(np.abs(decomp[0]))
        im = ax[0].imshow(cropped_decomp, cmap='RdBu_r', vmin=-clim, vmax=clim)
        ax[0].set_title('Spatial Filter')
        plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
        ax[1].plot(decomp[1][0])
        peak_frame = np.argmax(np.abs(decomp[1]))
        ax[1].set_title(f'Temporal Filter, Peak at frame {peak_frame}')
        output_dir = f'/home/zalaoui/higanbana/svgs'
        plt.show()

        





