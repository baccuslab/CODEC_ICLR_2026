import os
import numpy as np
import torch
import h5py as h5
import bscope
import torchdeepretina as tdr
from torch.utils.data import DataLoader, TensorDataset
import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Retinal model parameters
identifier = '15-11-21b_naturalscene'

print(f"Identifier: {identifier}")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Model and data paths
model_path = f"/home/zalaoui/torch-deep-retina/models/{identifier}.pt"
data_path = "/data/retina"
singletarget='surprisal' 
if 'surprisal' in singletarget:
    surprisal=True
else: 
    surpisal=False


# Output directory
output_dir = f'/home/zalaoui/retinal_codec/{identifier}_codec'
os.makedirs(output_dir, exist_ok=True)

print("Starting retinal contribution collection...")

# =============================================================================
# LOAD MODEL AND DATA
# =============================================================================

# Load retinal model
checkpt = tdr.io.load_checkpoint(model_path)
model = tdr.io.load_model(model_path)
model = tdr.utils.stacked2conv(model)  # Convert to regular Conv2d if needed
model.to(device)
model.eval()

print(f"Model architecture:\n{model}")

# Extract model parameters
dataset = checkpt['dataset'] 
cells = checkpt['cells']
stim_type = checkpt["stim_type"]
temporal_depth = checkpt['img_shape'][0]
norm_stats = [checkpt['norm_stats']['mean'], checkpt['norm_stats']['std']]

print(f"Dataset: {dataset}")
print(f"Cells: {cells}")
print(f"Stimulus type: {stim_type}")
print(f"Temporal depth: {temporal_depth}")

# Load test data using retinal data loader
test_data = tdr.datas.loadexpt(
    expt=dataset,
    cells=cells,
    filename=stim_type,
    train_or_test='test',
    history=temporal_depth,
    nskip=0,
    norm_stats=norm_stats,
    data_path=data_path
)

print(f"Test data shape: {test_data.X.shape}")
print(f"Test responses shape: {test_data.y.shape}")
print(f"Number of ganglion cells: {len(test_data.cells)}")

stimulus_tensor = torch.tensor(test_data.X).float()
response_tensor = torch.tensor(test_data.y).float()
dataset_torch = TensorDataset(stimulus_tensor, response_tensor)


test_loader = DataLoader(dataset_torch, batch_size=16, shuffle=False, 
                        num_workers=4, pin_memory=True)

print(f"Created DataLoader with {len(test_loader)} batches")

# =============================================================================
# SET UP BSCOPE FOR RETINAL LAYERS
# =============================================================================
print(f"\nSetting up bscope...")


print("Setting up bscope for retinal conv layers...")
print(model.sequential[0], model.sequential[4], model.sequential[8])

layers = [
    model.sequential[0],  # First conv layer
    model.sequential[4],  # Second conv layer  
    model.sequential[8]   # Final conv layer
]

# =============================================================================
# SINGLE TARGET - NO CELL LOOP NEEDED
# =============================================================================

# Initialize storage for firing rate sum contributions
all_contributions_by_layer = {}
all_activations_by_layer = {}
all_gradients_by_layer = {}
all_metadata = []
all_surprisal_data = []

# Initialize storage for each layer
layer_names = ['conv0', 'conv1', 'conv2']
for layer_name in layer_names:
    all_contributions_by_layer[layer_name] = []
    all_activations_by_layer[layer_name] = []
    all_gradients_by_layer[layer_name] = []


if surprisal:

    all_model_responses = []

    print(f" Collecting Statistics for Surprisal")
    for i, batch_data in enumerate(tqdm.tqdm(test_loader, desc="Surprisal Stats")):
        stimulus, response = batch_data
        stimulus = stimulus.to(device).float()
        model_output = model(stimulus)  # Forward pass to collect stats
        all_model_responses.append(model_output.detach().cpu().numpy())

    all_outputs = np.concatenate(all_model_responses, axis=0)


    # Compute Surprisal statistics
    mu = np.mean(all_outputs, axis=0)
    sigma = np.cov(all_outputs.T) 
    sigma_inv = np.linalg.inv(sigma)

mu_tensor = torch.from_numpy(mu).to(device).float() 
sigma_inv_tensor = torch.from_numpy(sigma_inv).to(device).float() 
print(f"Surprisal mu shape: {mu.shape}, sigma_inv shape: {sigma_inv.shape}, sigma: {sigma.shape}")
scope = bscope.Scope(model, layers)
scope.use_int_grad(steps=20)
scope.wrt_surprisal(softmax=False)  
scope.set_surprisal_stats(mu, sigma_inv)
scope.log_start(reduction=['ei_split', 'spatial_sum'])

# Run inference through batches 
all_responses = []
actual_responses = []
for i, batch_data in enumerate(tqdm.tqdm(test_loader, desc="Processing all cells")):
    stimulus, response = batch_data
    stimulus = stimulus.to(device).float()
    
    # Forward pass through model - outputs all cells
    output = model(stimulus)  # Shape: [batch_size, n_cells]
    


    
    centered = output - mu_tensor
    surprisal = 0.5 * torch.sum((centered @ sigma_inv_tensor) * centered, dim=1)
    all_surprisal_data.append(surprisal.detach().cpu().numpy())  

    scope(stimulus)
    
    # Store model predictions forv cells
    all_responses.append(output.detach().cpu().numpy()) 
    actual_responses.append(response.numpy())

scope.log_stop()

# Get responses for all cells and timepoints
model_responses = np.concatenate(all_responses, axis=0)  # [n_timepoints, n_cells]
actual_responses = np.concatenate(actual_responses, axis=0)  # [n_timepoints, n_cells]
all_surprisal_values = np.concatenate(all_surprisal_data, axis=0)  # [n_timepoints]



print(f"Processed {model_responses.shape[0]} timepoints for {model_responses.shape[1]} cells")
print(f"Model response range: {model_responses.min():.1f} - {model_responses.max():.1f} Hz")
print(f"Actual response range: {actual_responses.min():.1f} - {actual_responses.max():.1f} Hz")

# Calculate correlations per cell
n_cells = len(test_data.cells)
n_timepoints = model_responses.shape[0]
correlations = []
for cell_idx in range(n_cells):
    corr = np.corrcoef(model_responses[:, cell_idx], actual_responses[:, cell_idx])[0, 1]
    correlations.append(corr)
    print(f"Cell {test_data.cells[cell_idx]} correlation: {corr:.3f}")

# Store contributions for each layer 
for layer_idx, layer_name in enumerate(layer_names):
    layer_contributions = scope.log_contributions[layer_idx]
    layer_activations = scope.log_activations[layer_idx]
    layer_gradients = scope.log_gradients[layer_idx]
    print(f"Layer {layer_name} {singletarget} contributions shape: {layer_contributions.shape}")
    print(f"Layer {layer_name} activations shape: {layer_activations.shape}")
    print(f"Layer {layer_name} gradients shape: {layer_gradients.shape}")
    

    all_contributions_by_layer[layer_name] = layer_contributions
    all_activations_by_layer[layer_name] = layer_activations
    all_gradients_by_layer[layer_name] = layer_gradients

# Create metadata for all timepoints 
for timepoint_idx in range(n_timepoints):
    all_metadata.append({
        'timepoint': timepoint_idx,
        'model_responses_all_cells': model_responses[timepoint_idx, :],  
        'actual_responses_all_cells': actual_responses[timepoint_idx, :], 
        'scalar_target_sum': np.sum(model_responses[timepoint_idx, :])  
    })

# =============================================================================
# SAVE SINGLE TARGET contributions
# =============================================================================

print(f"\nSaving single target {singletarget} contributions...")
print(f"Total timepoints: {n_timepoints}")
print(f"Total cells: {n_cells}")

output_h5_path = os.path.join(output_dir, f'{identifier}_{singletarget}.h5')

with h5.File(output_h5_path, 'w') as f:
    # Save metadata
    metadata_group = f.create_group('metadata')
    metadata_group.create_dataset('timepoints', data=np.arange(n_timepoints))
    metadata_group.create_dataset('model_responses_all_cells', data=model_responses)  # [timepoints, cells]
    metadata_group.create_dataset('actual_responses_all_cells', data=actual_responses)  # [timepoints, cells]
    metadata_group.create_dataset('scalar_target_sums', data=np.array([m['scalar_target_sum'] for m in all_metadata]))
    metadata_group.create_dataset('surprisal_values', data=all_surprisal_values) 
    metadata_group.create_dataset('correlations_per_cell', data=np.array(correlations))
    
    # Save retinal-specific metadata
    metadata_group.create_dataset('cells_list', data=np.array(test_data.cells))
    metadata_group.create_dataset('layer_names', data=np.array(layer_names, dtype='S'))
    metadata_group.create_dataset('dataset', data=dataset.encode('utf-8'))
    metadata_group.create_dataset('stim_type', data=stim_type.encode('utf-8'))
    metadata_group.create_dataset('temporal_depth', data=temporal_depth)
    metadata_group.create_dataset('attribution_method', data=f'{singletarget}_single_target'.encode('utf-8'))
    metadata_group.create_dataset('n_cells', data=n_cells)
    metadata_group.create_dataset('n_timepoints', data=n_timepoints)
    
    # Save stimulus normalization stats
    metadata_group.create_dataset('norm_mean', data=norm_stats[0])
    metadata_group.create_dataset('norm_std', data=norm_stats[1])
    
    # Save single target firing rate sum contributions for each layer
    singletarget_group = f.create_group(f'{singletarget}_contributions')
    for layer_name in layer_names:
        contributions = all_contributions_by_layer[layer_name]
        singletarget_group.create_dataset(layer_name, data=contributions)

        
        # Add attributes
        singletarget_group[layer_name].attrs['shape_description'] = 'timepoints, spatial_dims'
        singletarget_group[layer_name].attrs['n_timepoints'] = n_timepoints
        singletarget_group[layer_name].attrs['attribution_method'] = f'{singletarget}_single_target'
        singletarget_group[layer_name].attrs['target_description'] = 'sum_all_ganglion_cells'
    
    # Save activations for each layer
    activations_group = f.create_group('activations')
    for layer_name in layer_names:
        activations = all_activations_by_layer[layer_name]
        activations_group.create_dataset(layer_name, data=activations)
        print(f"Saved activations {layer_name}: {activations.shape} (timepoints, channels/spatial)")
        activations_group[layer_name].attrs['shape_description'] = 'timepoints, channels_spatial'
        activations_group[layer_name].attrs['n_timepoints'] = n_timepoints
    
    # Save gradients for each layer
    gradients_group = f.create_group('gradients')
    for layer_name in layer_names:
        gradients = all_gradients_by_layer[layer_name]
        gradients_group.create_dataset(layer_name, data=gradients)
        print(f"Saved gradients {layer_name}: {gradients.shape} (timepoints, channels/spatial)")
        gradients_group[layer_name].attrs['shape_description'] = 'timepoints, channels_spatial'
        gradients_group[layer_name].attrs['n_timepoints'] = n_timepoints

print(f"\nSaved single target {singletarget} contributions to: {output_h5_path}")

