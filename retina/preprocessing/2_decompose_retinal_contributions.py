import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import h5py as h5
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt
import bscope
from bscope import r2_score




# =============================================================================
# CONFIGURATION FOR SINGLE TARGET SAE
# =============================================================================


identifier = '15-11-21b_naturalscene'
saveflag='TEST'  
singletarget = 'surprisal'  
data_path = f'/home/zalaoui/retinal_codec/{identifier}_codec/{identifier}_{singletarget}.h5'

# SAE parameters
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128  
N_DICT_MULTIPLIER = 10 
THRESHOLD = 0.9
LEARNING_RATE = 1e-5
EPOCHS = 800
CODE_L1 = None
ATOM_L1 = 1e-4



sae_output_dir = f'/home/zalaoui/retinal_codec/{identifier}_codec/single_target_saes'
os.makedirs(sae_output_dir, exist_ok=True)



# =============================================================================
# LOAD SINGLE TARGET DATA
# =============================================================================

with h5.File(data_path, 'r') as f:
    # Load metadata
    cells_list = f['metadata/cells_list'][:]
    layer_names = [name.decode('utf-8') for name in f['metadata/layer_names'][:]]
    model_responses_all_cells = f['metadata/model_responses_all_cells'][:]  # [timepoints, cells]
    actual_responses_all_cells = f['metadata/actual_responses_all_cells'][:]  # [timepoints, cells]
    scalar_target_sums = f['metadata/scalar_target_sums'][:]  # [timepoints]
    n_cells = f['metadata/n_cells'][()]
    n_timepoints = f['metadata/n_timepoints'][()]
    

    contributions_by_layer = {}
    activations_by_layer = {}
    gradients_by_layer = {}
    
    for layer_name in layer_names:
        contributions_by_layer[layer_name] = f[f'{singletarget}_contributions/{layer_name}'][:]
        activations_by_layer[layer_name] = f[f'activations/{layer_name}'][:]
        gradients_by_layer[layer_name] = f[f'gradients/{layer_name}'][:]
        print(f"Loaded {layer_name} contributions: {contributions_by_layer[layer_name].shape}")
        print(f"Loaded {layer_name} activations: {activations_by_layer[layer_name].shape}")

print(f"Data loaded: {n_cells} cells, {n_timepoints} timepoints")
print(f"Target: Total network activity ({singletarget})")
print(f"Layers: {layer_names}")

# =============================================================================
# TRAIN SAE FOR EACH LAYER ON SINGLE TARGET CONTRIBUTIONS
# =============================================================================

# Storage for trained SAEs and results
single_target_saes = {}
single_target_results = {}

for layer_name in layer_names:

    print(f"\n{'='*60}")
    print(f"PROCESSING LAYER: {layer_name} - SINGLE TARGET CONTRIBUTIONS")
    print(f"{'='*60}")
    
    # Get contributions for this layer: [timepoints, spatial_dims]
    layer_contribs = contributions_by_layer[layer_name]
    layer_activs = activations_by_layer[layer_name] 
    layer_grads = gradients_by_layer[layer_name]
    
    print(f"Layer {layer_name}:")
    print(f"  Contributions shape: {layer_contribs.shape}")
    print(f"  Activations shape: {layer_activs.shape}")
    print(f"  Gradients shape: {layer_grads.shape}")
    

    n_features = layer_contribs.shape[1]
    data_for_sae = layer_contribs  # [timepoints, features]
    print(f"  Data for SAE shape: {data_for_sae.shape}")
    print(f"  Number of features: {n_features}")

    from sklearn.decomposition import PCA

    # For each layer, check intrinsic dimensionality
    pca = PCA(n_components=data_for_sae.shape[1])
    pca.fit(data_for_sae)
    explained_var_ratio = pca.explained_variance_ratio_
    cumsum = np.cumsum(explained_var_ratio)

    n_components_90 = np.argmax(cumsum >= 0.9) + 1
    print(f"PCA: {n_components_90} components explain 90% variance")

    print(f"  Input features for SAE: {n_features}")
    print(f"  Dictionary size: {n_features * N_DICT_MULTIPLIER}")
    print(f"  Training samples: {data_for_sae.shape[0]}")
    

    data_mean = data_for_sae.mean()
    data_std = data_for_sae.std()
    data_normalized = data_for_sae / data_std

    
    # Create PyTorch datasets
    data_tensor = torch.from_numpy(data_normalized).float()
    dataset = TensorDataset(data_tensor)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )
    
    eval_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True,
        num_workers=4
    )
    
    # Initialize SAE for this layer
    sae = bscope.STSAE(
        n_features,
        num_atoms=n_features * N_DICT_MULTIPLIER,
        threshold=THRESHOLD,
        mlp_hidden_dim=data_for_sae.shape[-1] # Minimum hidden dim
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.98)
    # (optimizer, patience=75, factor=0.95)
    loss_fn = nn.MSELoss()
    

    
    # Training loop
    best_r2 = -np.inf
    best_sae_state = None
    
    for epoch in range(EPOCHS):
        sae.train()
        loss_agg=0
        for batch in train_loader:
            batch_data = batch[0].to(DEVICE)
            optimizer.zero_grad()
            
            pre_codes, codes, reconstructed = sae(batch_data)
            
            # Reconstruction loss
            loss = loss_fn(reconstructed, batch_data)
            

            
            # Sparsity regularization
            if CODE_L1 is not None:
                loss += CODE_L1 * torch.mean(torch.abs(codes))
            if ATOM_L1 is not None:
                loss += ATOM_L1 * torch.mean(torch.abs(sae.dictionary.get_dictionary()))
            loss.backward()
            loss_agg += loss.item()
            optimizer.step()
            
        losses = loss_agg / len(train_loader)
        scheduler.step()

        
        # Evaluation every 25 epochs
        if epoch % 25 == 0 or epoch == EPOCHS - 1:
            sae.eval()
            all_eval_data = []
            all_eval_reconstructed = []
            all_eval_codes = []
            
            with torch.no_grad():
                for batch in eval_loader:
                    batch_data = batch[0].to(DEVICE)
                    pre_codes, codes, reconstructed = sae(batch_data)
                    
                    all_eval_data.append(batch_data.cpu().numpy())
                    all_eval_reconstructed.append(reconstructed.cpu().numpy())
                    all_eval_codes.append(codes.cpu().numpy())
            
            # Calculate reconstruction metrics
            eval_data = np.concatenate(all_eval_data, axis=0)
            eval_recon = np.concatenate(all_eval_reconstructed, axis=0)
            eval_codes = np.concatenate(all_eval_codes, axis=0)


            r2 = r2_score(torch.from_numpy(eval_data), torch.from_numpy(eval_recon))
            
            # Calculate sparsity metrics
            code_activity = (eval_codes > 0).mean(axis=0)  # Fraction of time each atom is active
            active_atoms = (code_activity > 0.01).sum()  # Atoms active >1% of time
            mean_l0 = (eval_codes > 0).sum(axis=1).mean()  # Average atoms per timepoint

            # ADD ALIVE/DEAD ANALYSIS HERE
            firings = np.sum(np.abs(eval_codes), axis=0)
            alive_mask = firings > 0
            n_alive = alive_mask.sum()
            n_dead = (~alive_mask).sum()


            print(f"  Epoch {epoch+1:3d}: Loss={losses:.4f}, R²={r2:.4f}, "
                f"Active atoms={active_atoms}/{eval_codes.shape[1]}, Mean L0={mean_l0:.1f}, "
                f"Alive={n_alive}, Dead={n_dead}, "
                f"LR={optimizer.param_groups[0]['lr']:.2e}")
                

            firings = np.sum(np.abs(eval_codes), axis=0)  # Use eval_codes
            alive_mask = firings > 0  # Boolean mask [n_atoms]

            # Get the full dictionary first
            full_dictionary = sae.dictionary.get_dictionary().detach().cpu().numpy()
            alive_dictionary = full_dictionary[alive_mask]  # Now correctly indexed

            n_alive = alive_mask.sum()
            n_dead = (~alive_mask).sum()


            if r2 > best_r2:
                best_r2 = r2
                best_sae_state = sae.state_dict().copy()
    
    print(f"Final R² for layer {layer_name}: {best_r2:.4f}")
    
    # Load best model state
    if best_sae_state is not None:
        sae.load_state_dict(best_sae_state)
    
    # Store trained SAE
    single_target_saes[layer_name] = sae.cpu()
    
    # Final evaluation with best model
    sae.eval()
    with torch.no_grad():
        sae = sae.to(DEVICE)  # Move SAE back to GPU
        data_tensor_gpu = torch.from_numpy(data_normalized).float().to(DEVICE)
        pre_codes, final_codes, final_reconstructed = sae(data_tensor_gpu)
        
        final_codes_np = final_codes.cpu().numpy()
        final_reconstructed_np = final_reconstructed.cpu().numpy()

    firings = np.sum(np.abs(final_codes_np), axis=0)
    alive_mask = firings > 0
    alive_codes = final_codes_np[:, alive_mask]
    alive_dictionary = sae.dictionary.get_dictionary().detach().cpu().numpy()[alive_mask]
    n_alive = alive_mask.sum()
    n_dead = (~alive_mask).sum()
    print(f"Layer {layer_name} - Alive atoms: {n_alive}, Dead atoms: {n_dead}")
    
    # Denormalize reconstruction
    final_reconstructed_denorm = final_reconstructed_np * data_std + data_mean
    
    # Store results
    single_target_results[layer_name] = {
        'original_attributions': data_for_sae,
        'normalized_attributions': data_normalized,
        'codes': alive_codes,
        'reconstructed_normalized': final_reconstructed_np,
        'reconstructed_denormalized': final_reconstructed_denorm,
        'dictionary': alive_dictionary,
        'normalization_stats': {'mean': data_mean, 'std': data_std},
        'r2_score': best_r2,
        'active_atoms': active_atoms,
        'mean_l0': mean_l0,
        'code_activity': code_activity,
        'n_features': n_features,
        'n_atoms': n_features * N_DICT_MULTIPLIER
    }
    
    # Save individual SAE
    sae_save_path = os.path.join(sae_output_dir, f'{layer_name}_{singletarget}_sae_{saveflag}.pt')
    torch.save(sae.cpu(), sae_save_path)
    print(f"Saved SAE: {sae_save_path}")

# =============================================================================
# SAVE RESULTS TO HDF5
# =============================================================================

results_h5_path = os.path.join(sae_output_dir, f'{singletarget}_sae_results_{saveflag}.h5')
print(f"\nSaving single target SAE results to: {results_h5_path}")

with h5.File(results_h5_path, 'w') as f:
    # Save metadata
    meta_group = f.create_group('metadata')
    meta_group.create_dataset('cells_list', data=np.array(cells_list))
    meta_group.create_dataset('layer_names', data=np.array(layer_names, dtype='S'))
    meta_group.create_dataset('n_dict_multiplier', data=N_DICT_MULTIPLIER)
    meta_group.create_dataset('identifier', data=identifier.encode('utf-8'))
    meta_group.create_dataset('n_cells', data=n_cells)
    meta_group.create_dataset('n_timepoints', data=n_timepoints)
    meta_group.attrs['target_type'] = 'total_network_activity'
    meta_group.attrs['attribution_method'] = f'{singletarget}_single_target'
    
    # Save target activity data
    targets_group = f.create_group('target_data')
    targets_group.create_dataset('scalar_target_sums', data=scalar_target_sums)
    targets_group.create_dataset('model_responses_all_cells', data=model_responses_all_cells)
    targets_group.create_dataset('actual_responses_all_cells', data=actual_responses_all_cells)
    
    # Save SAE results for each layer
    for layer_name in layer_names:
        if layer_name in single_target_results:
            layer_group = f.create_group(layer_name)
            results = single_target_results[layer_name]
            
            # Save all arrays
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    layer_group.create_dataset(key, data=value)
                elif isinstance(value, dict):
                    # Handle nested dict (normalization_stats)
                    subgroup = layer_group.create_group(key)
                    for subkey, subvalue in value.items():
                        subgroup.create_dataset(subkey, data=subvalue)
                else:
                    layer_group.attrs[key] = value

print(f"Saved single target SAE results for {len(single_target_results)} layers")

