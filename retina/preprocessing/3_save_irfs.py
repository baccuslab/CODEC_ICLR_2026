import os
import numpy as np
import torch
import torchdeepretina as tdr
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import h5py as h5
import bscope


def batch_irf(model, stim, neuron_index, device):
    """Standard gradient-based IRF computation"""
    if isinstance(stim, np.ndarray):
        stim = torch.tensor(stim, dtype=torch.float32, device=device)

    if stim.ndim == 3:
        stim = stim.unsqueeze(0)
    
    stim.requires_grad_(True)
    model.zero_grad()

    if stim.grad is not None:
        stim.grad.zero_()
    
    output = model(stim)
    neuron_activation = output[:, neuron_index]
    scalar_output = neuron_activation.sum()
    scalar_output.backward()
    
    gradients = stim.grad.detach()
    return gradients

def compute_and_save_irfs(identifier, selected_cells=None, ig_steps=2, output_dir=None,compute_ig=False):
    """
    Compute both IG-IRF and regular IRF for specified cells and save to HDF5.
    
    Args:
        identifier: Model identifier (e.g., '15-10-07_naturalscene')
        selected_cells: List of cell IDs to process. If None, processes all cells.
        ig_steps: Number of interpolation steps for integrated gradients
        output_dir: Directory to save results. If None, uses current directory.
    """
    
    # Setup paths and device
    model_path = f"/home/zalaoui/torch-deep-retina/models/{identifier}.pt"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Processing model: {identifier}")
    
    # Load model and data
    checkpt = tdr.io.load_checkpoint(model_path)
    model = tdr.io.load_model(model_path)
    model = tdr.utils.stacked2conv(model)
    model.to(device)
    model.eval()
    
    # Extract parameters
    dataset = checkpt['dataset'] 
    cells = checkpt['cells']
    stim_type = checkpt["stim_type"]
    window_size = checkpt['img_shape'][0]
    height = checkpt['img_shape'][1]
    width = checkpt['img_shape'][2]
    norm_stats = [checkpt['norm_stats']['mean'], checkpt['norm_stats']['std']]
    
    print(f"Dataset: {dataset}, Cells: {len(cells)}")
    print(f"Stimulus: {stim_type}, Dims: ({window_size}, {height}, {width})")
    
    # Load test data
    test_data = tdr.datas.loadexpt(dataset, cells, stim_type, 'test',
                                  window_size, nskip=0,
                                  norm_stats=norm_stats,
                                  data_path="/data/retina")
    print(f"cells type: {type(cells[0])}, value: {cells[0]}")
    print(f"test_data.cells type: {type(test_data.cells[0])}, value: {test_data.cells[0]}")
    # Convert to tensors and create dataloader
    stimulus_tensor = torch.tensor(test_data.X).float()
    response_tensor = torch.tensor(test_data.y).float()
    dataset_torch = TensorDataset(stimulus_tensor, response_tensor)
    test_loader = DataLoader(dataset_torch, batch_size=1, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Determine which cells to process
    if selected_cells is None:
        selected_cells = test_data.cells
        
        print(" Selecting all cells.")
        print(f" Cells: {selected_cells}")
    else:
        # Verify all selected cells exist
        missing_cells = [c for c in selected_cells if c not in cells]
        if missing_cells:
            print(f"Warning: Cells {missing_cells} not found in model. Skipping.")
            selected_cells = [c for c in selected_cells if c in cells]
    
    print(f"Processing {len(selected_cells)} cells: {selected_cells}")
    
    # Prepare output file
    output_file = os.path.join(output_dir, f"{identifier}_irfs.h5")
    print(f"Saving results to: {output_file}")
    
    # Setup bscope for IG-IRF
    layers = [model.sequential[0]]  # Layer doesn't matter for input gradients
    if compute_ig:
        scope = bscope.Scope(model, layers)
        scope.use_input_int_grad(steps=ig_steps)
    
    with h5.File(output_file, 'w') as f:
        # Save metadata
        metadata_grp = f.create_group('metadata')
        metadata_grp.attrs['identifier'] = identifier
        metadata_grp.attrs['dataset'] = dataset
        metadata_grp.attrs['stim_type'] = stim_type
        metadata_grp.attrs['window_size'] = window_size
        metadata_grp.attrs['height'] = height
        metadata_grp.attrs['width'] = width
        if compute_ig:
            metadata_grp.attrs['ig_steps'] = ig_steps
        metadata_grp.attrs['n_samples'] = len(test_loader)
        metadata_grp.create_dataset('cells_processed', data=selected_cells)
        metadata_grp.create_dataset('all_cells', data=cells)
        metadata_grp.create_dataset('norm_stats_mean', data=norm_stats[0])
        metadata_grp.create_dataset('norm_stats_std', data=norm_stats[1])
        
        # Process each cell
        for cell_id in selected_cells:
            cell_idx = selected_cells.index(cell_id)
            print(f"\nProcessing Cell {cell_id} (index {cell_idx})...")
            
            # Create group for this cell
            cell_grp = f.create_group(f'cell_{cell_id}')
            cell_grp.attrs['cell_id'] = cell_id
            cell_grp.attrs['cell_index'] = cell_idx
            

            if compute_ig:
                scope.wrt_output_neuron(neuron_index=cell_idx, softmax=False)
                scope.log_start()
                print("Computing IG-IRF...")
            else:
                print("Computing regular IRF only...")

            ig_gradients = []
            regular_gradients = []

            for i, batch_data in enumerate(tqdm.tqdm(test_loader, desc=f"Cell {cell_id}")):
                stimulus, response = batch_data
                stimulus = stimulus.to(device).float()
                
                # Compute IG-IRF only if flag is set
                if compute_ig:
                    scope(stimulus)
                    ig_grad = scope.contributions[0]
                    ig_gradients.append(ig_grad)
                
                # Always compute regular IRF
                regular_grad = batch_irf(model, stimulus, neuron_index=cell_idx, device=device)
                regular_gradients.append(regular_grad.cpu().numpy())
                
                if i % 100 == 0:
                    torch.cuda.empty_cache()

            if compute_ig:
                scope.log_stop()
                ig_all = scope.log_contributions[0]
                ig_mean = ig_all.mean(axis=0)
                cell_grp.create_dataset('ig_irf_all', data=ig_all, compression='gzip')
                cell_grp.create_dataset('ig_irf_mean', data=ig_mean, compression='gzip')
                cell_grp.attrs['ig_irf_min'] = float(ig_all.min())
                cell_grp.attrs['ig_irf_max'] = float(ig_all.max())

            # Always save regular IRF
            regular_all = np.concatenate(regular_gradients, axis=0)
            regular_mean = regular_all.mean(axis=0)
            cell_grp.create_dataset('regular_irf_all', data=regular_all, compression='gzip')
            cell_grp.create_dataset('regular_irf_mean', data=regular_mean, compression='gzip')
            cell_grp.attrs['regular_irf_min'] = float(regular_all.min())
            cell_grp.attrs['regular_irf_max'] = float(regular_all.max())
            
            print(f"Saved IRFs for Cell {cell_id}")
    
    print(f"\nCompleted! Results saved to: {output_file}")
    return output_file

def load_irfs(file_path, cell_id, load_ig_irf=False):
    """
    Load IRF data for a specific cell from the saved HDF5 file.

    Args:
        file_path: Path to the HDF5 file
        cell_id: ID of the cell to load
        load_ig_irf: if True, also load ig_irf_mean and ig_irf_all (default True);
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
        


identifier = '15-11-21b_naturalscene'
selected_cells = None
ig_steps = 5
output_dir = f'/home/zalaoui/retina_codec/{identifier}/irf_results'

# Run computation
output_file = compute_and_save_irfs(
    identifier=identifier,
    selected_cells=selected_cells,
    ig_steps=ig_steps,
    output_dir=output_dir, compute_ig=False
)
    
