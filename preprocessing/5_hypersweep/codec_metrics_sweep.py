import h5py as h5
import bscope
import bscope.ic as bic
import glob
import os
import torch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from bscope.metrics import *
from torch.utils.data import TensorDataset

# Configuration
experiment = 'resnet50_act_normgrad_top_3_nosoftmax'
exp_path = f'/data/codec/decomps/{experiment}/'
datatype = 'contributions'
layer = '15'
sweep_id = 'sae-sigthresh-sweep'
SPATIAL_OPERATION = 'sum'
mode_summary = True
path = f'/data/codec/decomps/{experiment}/saes/{sweep_id}/layer_{layer}/'
device = 'cuda:3'

# Your sweep configurations to compare (matches your current sweep)
sweep_configs = [
    {'threshold': 0.5, 'n': 1, 'atom_l1': 0},
    {'threshold': 0.5, 'n': 1, 'atom_l1': 1e-4},
    {'threshold': 0.5, 'n': 3, 'atom_l1': 0},
    {'threshold': 0.5, 'n': 3, 'atom_l1': 1e-4},
    {'threshold': 0.5, 'n': 8, 'atom_l1': 0},
    {'threshold': 0.5, 'n': 8, 'atom_l1': 1e-4},
    {'threshold': 0.7, 'n': 1, 'atom_l1': 0},
    {'threshold': 0.7, 'n': 1, 'atom_l1': 1e-4},
    {'threshold': 0.7, 'n': 3, 'atom_l1': 0},
    {'threshold': 0.7, 'n': 3, 'atom_l1': 1e-4},
    {'threshold': 0.7, 'n': 8, 'atom_l1': 0},
    {'threshold': 0.7, 'n': 8, 'atom_l1': 1e-4},
    {'threshold': 0.9, 'n': 3, 'atom_l1': 0},
    {'threshold': 0.9, 'n': 3, 'atom_l1': 1e-4},
    {'threshold': 0.9, 'n': 8, 'atom_l1': 0},
    {'threshold': 0.9, 'n': 8, 'atom_l1': 1e-4},
]

datapath = os.path.join(exp_path, 'data.h5')

# Load the data once (same for all SAEs) - match your training script exactly
contributions, targets = bic.load_contribution_data(datapath, datatype, layer, SPATIAL_OPERATION)

# Initialize results list
results = []

print("Evaluating SAEs from sweep...")
print("=" * 50)

for config in sweep_configs:
    threshold = config['threshold']
    n = config['n']
    atom_l1 = config['atom_l1']
    
    if atom_l1 == 1e-4:
        atom_l1_formatted = "0.0001"
    elif atom_l1 == 0:
        atom_l1_formatted = "0"
    else:
        atom_l1_formatted = str(atom_l1)

    run_dir = os.path.join(path, f"hypersweep_{threshold}_{n}_{atom_l1_formatted}")
    sae_identifier = f"{threshold}_{n}_{atom_l1_formatted}"
    
    try:
        if mode_summary:
            mode_summary_path = os.path.join(run_dir, 'mode_summary.h5')
            
            if not os.path.exists(mode_summary_path):
                print(f"✗ Mode summary not found for {sae_identifier}: {mode_summary_path}")
                # Add failed metrics
                failed_metrics = {
                    'SAE_ID': sae_identifier,
                    'Threshold': threshold,
                    'N': n,
                    'ATOM_L1': atom_l1,
                    'R²': np.nan,
                    'MSE': np.nan,
                    'L1_Loss': np.nan,
                    'Relative_L2': np.nan,
                    'Dead_Codes': np.nan,
                    'L0': np.nan,
                    'L0_eps': np.nan,
                    'L1_L2_Ratio': np.nan,
                    'Hoyer': np.nan,
                    'Kappa_4': np.nan,
                    'Max_Cosine': np.nan,
                    'Coherence': np.nan,
                    'Stable_Rank': np.nan,
                    'Eff_Rank': np.nan,
                    'Connectivity': np.nan,
                    'Neg_Inter': np.nan,
                    'OOD_Score': np.nan,
                    'Stability': np.nan,
                    'Num_Alive_Codes': np.nan,
                    'Num_Dead_Codes': np.nan,
                    'Dict_Norm_Min': np.nan,
                    'Dict_Norm_Max': np.nan,
                }
                results.append(failed_metrics)
                continue
                
            m = bic.ModeSummary(mode_summary_path)
            
            # Load the data for that layer
            corr_mtx = m.file[f'layers/{layer}/corr_mtx'][:]
            r2 = m.file[f'layers/{layer}/r2'][()]  # Scalar
            codes = torch.from_numpy(m.file[f'layers/{layer}/loadings'][:]).float()
            dictionary = torch.from_numpy(m.file[f'layers/{layer}/dictionary'][:]).float()
            data_tensor = torch.from_numpy(m.file[f'layers/{layer}/data_agg'][:]).float()
            reconstructed = torch.from_numpy(m.file[f'layers/{layer}/reconstructed_agg'][:]).float()
            
            # Close the HDF5 file
            m.file.close()
            
        else: 
            # Look for the full model first
            full_model_path = os.path.join(run_dir, 'sae_full_model_epoch_300.pt')
            
            if not os.path.exists(full_model_path):
                print(f"✗ No model found for {sae_identifier}")
                # Add failed metrics
                failed_metrics = {
                    'SAE_ID': sae_identifier,
                    'Threshold': threshold,
                    'N': n,
                    'ATOM_L1': atom_l1,
                    'R²': np.nan,
                    'MSE': np.nan,
                    'L1_Loss': np.nan,
                    'Relative_L2': np.nan,
                    'Dead_Codes': np.nan,
                    'L0': np.nan,
                    'L0_eps': np.nan,
                    'L1_L2_Ratio': np.nan,
                    'Hoyer': np.nan,
                    'Kappa_4': np.nan,
                    'Max_Cosine': np.nan,
                    'Coherence': np.nan,
                    'Stable_Rank': np.nan,
                    'Eff_Rank': np.nan,
                    'Connectivity': np.nan,
                    'Neg_Inter': np.nan,
                    'OOD_Score': np.nan,
                    'Stability': np.nan,
                    'Num_Alive_Codes': np.nan,
                    'Num_Dead_Codes': np.nan,
                    'Dict_Norm_Min': np.nan,
                    'Dict_Norm_Max': np.nan,
                }
                results.append(failed_metrics)
                continue
            
            # Load using bscope.load_sae with the actual data
            out = bscope.load_sae(full_model_path, contributions, device)
            sae = out[0]
            codes = torch.from_numpy(out[1]).float()
            dictionary = torch.from_numpy(out[2]).float()
            data_tensor = torch.from_numpy(out[3]).float()
            reconstructed = torch.from_numpy(out[4]).float()
            r2 = out[5]

        # Now compute metrics (this applies to both mode_summary and full model paths)
        metrics = {}
        metrics['SAE_ID'] = sae_identifier
        metrics['Threshold'] = threshold
        metrics['N'] = n
        metrics['ATOM_L1'] = atom_l1
        
        # ===== BASIC RECONSTRUCTION METRICS =====
        metrics['R²'] = float(r2) if isinstance(r2, (np.ndarray, np.number)) else r2
        metrics['MSE'] = float(avg_l2_loss(data_tensor, reconstructed))
        metrics['L1_Loss'] = float(avg_l1_loss(data_tensor, reconstructed))
        metrics['Relative_L2'] = float(relative_avg_l2_loss(data_tensor, reconstructed))
        
        # ===== SPARSITY METRICS =====
        metrics['Dead_Codes'] = float(dead_codes(codes).mean())
        metrics['L0'] = float(l0(codes))
        metrics['L0_eps'] = float(l0_eps(codes))
        metrics['L1_L2_Ratio'] = float(l1_l2_ratio(codes).mean())
        metrics['Hoyer'] = float(hoyer(codes).mean())
        metrics['Kappa_4'] = float(kappa_4(codes).mean())
        
        # ===== DICTIONARY QUALITY METRICS =====
        max_collinearity, cosine_sim_matrix = dictionary_collinearity(dictionary)
        metrics['Max_Cosine'] = float(max_collinearity)
        metrics['Coherence'] = float(max_collinearity)  # Same as Max Cosine
        
        # ===== STABILITY AND RANKING METRICS =====
        # Stable Rank (effective rank based on singular values)
        U, S, V = torch.svd(dictionary)
        stable_rank = float((S.sum() ** 2) / (S ** 2).sum())
        metrics['Stable_Rank'] = stable_rank
        
        # Effective Rank (number of significant singular values)
        S_normalized = S / S.max()
        eff_rank = float((S_normalized > 0.01).sum())  # Count SVs > 1% of max
        metrics['Eff_Rank'] = eff_rank
        
        # ===== CONNECTIVITY AND INTERACTION METRICS =====
        # Connectivity (fraction of codes that interact)
        if codes.shape[1] > 1:  # Need at least 2 features for correlation
            code_correlations = torch.corrcoef(codes.T)
            code_correlations[torch.isnan(code_correlations)] = 0
            connectivity = float((torch.abs(code_correlations) > 0.1).float().mean())
            metrics['Connectivity'] = connectivity
            
            # Negative Interference (how much codes interfere negatively)
            negative_corrs = code_correlations[code_correlations < -0.1]
            metrics['Neg_Inter'] = float(torch.abs(negative_corrs).mean()) if len(negative_corrs) > 0 else 0.0
        else:
            metrics['Connectivity'] = 0.0
            metrics['Neg_Inter'] = 0.0
        
        # ===== OUT-OF-DISTRIBUTION SCORE =====
        data_points = torch.from_numpy(contributions).float()

        # Normalize dictionary and data for cosine similarity
        dict_normalized = dictionary / (dictionary.norm(dim=1, keepdim=True) + 1e-8)
        data_normalized = data_points / (data_points.norm(dim=1, keepdim=True) + 1e-8)

        # For each dictionary atom, find max cosine similarity with any data point
        max_similarities = torch.max(torch.mm(dict_normalized, data_normalized.T), dim=1)[0]

        # OOD Score: 1 - average of max similarities
        ood_score = 1 - torch.mean(max_similarities)
        metrics['OOD_Score'] = float(ood_score)
        
        # ===== STABILITY SCORE =====
        # Stability (consistency across different data batches)
        # ADD STABILLITY IF WE HAVE MULTIPLE RUNS
        metrics['Stability'] = 0

        # run_dirs = glob.glob(os.path.join(path, f"hypersweep_{threshold}_{n}_{atom_l1_formatted}_run*"))

        # if len(run_dirs) >= 2:
        #     dictionaries = []
        #     for run_dir in run_dirs[:2]:  # Use first 2 runs
        #         if mode_summary:
        #             ms_path = os.path.join(run_dir, 'mode_summary.h5')
        #             if os.path.exists(ms_path):
        #                 with h5.File(ms_path, 'r') as f:
        #                     dict_tensor = torch.from_numpy(f[f'layers/{layer}/dictionary'][:]).float()
        #                     dictionaries.append(dict_tensor)
        #         # ... similar for full model path
            
        #     if len(dictionaries) == 2:
        #         # Compute stability using Hungarian matching
        #         stability_score = compute_stability(dictionaries[0], dictionaries[1])
        #         metrics['Stability'] = float(stability_score)
        #     else:
        #         metrics['Stability'] = np.nan
        # else:
        #     metrics['Stability'] = np.nan
        
        # Additional metrics
        firings = codes.sum(0)
        alive = firings > 0
        metrics['Num_Alive_Codes'] = int(alive.sum())
        metrics['Num_Dead_Codes'] = int((~alive).sum())
        metrics['Dict_Norm_Min'] = float(dictionary.norm(dim=1).min())
        metrics['Dict_Norm_Max'] = float(dictionary.norm(dim=1).max())
        
        # Add checkpoint info

        
        results.append(metrics)
        print(f"✓ {sae_identifier} evaluated successfully")
        
    except Exception as e:
        print(f"✗ Error evaluating {sae_identifier}: {str(e)}")
        # Add a row with NaN values for failed evaluations
        failed_metrics = {
            'SAE_ID': sae_identifier,
            'Threshold': threshold,
            'N': n,
            'ATOM_L1': atom_l1,
            'R²': np.nan,
            'MSE': np.nan,
            'L1_Loss': np.nan,
            'Relative_L2': np.nan,
            'Dead_Codes': np.nan,
            'L0': np.nan,
            'L0_eps': np.nan,
            'L1_L2_Ratio': np.nan,
            'Hoyer': np.nan,
            'Kappa_4': np.nan,
            'Max_Cosine': np.nan,
            'Coherence': np.nan,
            'Stable_Rank': np.nan,
            'Eff_Rank': np.nan,
            'Connectivity': np.nan,
            'Neg_Inter': np.nan,
            'OOD_Score': np.nan,
            'Stability': np.nan,
            'Num_Alive_Codes': np.nan,
            'Num_Dead_Codes': np.nan,
            'Dict_Norm_Min': np.nan,
            'Dict_Norm_Max': np.nan,
        }
        results.append(failed_metrics)

# Create DataFrame
df = pd.DataFrame(results)

# Display results
print("\n" + "="*80)
print("SAE SWEEP COMPARISON RESULTS")
print("="*80)

# Show key metrics in a nice format
key_metrics = ['SAE_ID', 'Threshold', 'N', 'ATOM_L1', 'R²', 'MSE', 'L0', 'Num_Alive_Codes', 'Max_Cosine', 'Stable_Rank', 'Stability']
print("\nKey Metrics Summary:")
print(df[key_metrics].round(4).to_string(index=False))

print("\nFull Results:")
print(df.round(4).to_string(index=False))

# Save to CSV
output_file = f'sae_sweep_comparison_{experiment}_layer_{layer}.csv'
df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")

# Create comparison plots grouped by threshold and N
if len(results) > 1:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Clean data for plotting
    df_clean = df.dropna()
    
    if len(df_clean) > 0:
        # R² comparison
        x_pos = np.arange(len(df_clean))
        bars = axes[0,0].bar(x_pos, df_clean['R²'])
        axes[0,0].set_title('R² Comparison')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels([f"T{row['Threshold']}_N{row['N']}_A{row['ATOM_L1']}" for _, row in df_clean.iterrows()], rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, df_clean['R²']):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                          f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # L0 (sparsity) comparison
        bars = axes[0,1].bar(x_pos, df_clean['L0'])
        axes[0,1].set_title('L0 (Sparsity) Comparison')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels([f"T{row['Threshold']}_N{row['N']}_A{row['ATOM_L1']}" for _, row in df_clean.iterrows()], rotation=45)
        
        # Alive codes comparison
        bars = axes[0,2].bar(x_pos, df_clean['Num_Alive_Codes'])
        axes[0,2].set_title('Number of Alive Codes')
        axes[0,2].set_xticks(x_pos)
        axes[0,2].set_xticklabels([f"T{row['Threshold']}_N{row['N']}_A{row['ATOM_L1']}" for _, row in df_clean.iterrows()], rotation=45)
        
        # Max cosine similarity comparison
        bars = axes[1,0].bar(x_pos, df_clean['Max_Cosine'])
        axes[1,0].set_title('Max Cosine Similarity')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels([f"T{row['Threshold']}_N{row['N']}_A{row['ATOM_L1']}" for _, row in df_clean.iterrows()], rotation=45)
        
        # Stable rank comparison
        bars = axes[1,1].bar(x_pos, df_clean['Stable_Rank'])
        axes[1,1].set_title('Stable Rank Comparison')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels([f"T{row['Threshold']}_N{row['N']}_A{row['ATOM_L1']}" for _, row in df_clean.iterrows()], rotation=45)
        
        # Stability comparison
        bars = axes[1,2].bar(x_pos, df_clean['OOD_Score'])
        axes[1,2].set_title('OOD Comparison')
        axes[1,2].set_xticks(x_pos)
        axes[1,2].set_xticklabels([f"T{row['Threshold']}_N{row['N']}_A{row['ATOM_L1']}" for _, row in df_clean.iterrows()], rotation=45)
    
    plt.tight_layout()
    plt.show()

print(f"\nEvaluation complete! Processed {len(sweep_configs)} sweep configurations.")

# Show best performing models
if len(df_clean) > 0:
    print("\n" + "="*50)
    print("TOP PERFORMERS")
    print("="*50)
    
    print(f"\nBest R²: {df_clean.loc[df_clean['R²'].idxmax(), 'SAE_ID']} ({df_clean['R²'].max():.4f})")
    print(f"Most Sparse (lowest L0): {df_clean.loc[df_clean['L0'].idxmin(), 'SAE_ID']} ({df_clean['L0'].min():.2f})")
    print(f"Most Alive Codes: {df_clean.loc[df_clean['Num_Alive_Codes'].idxmax(), 'SAE_ID']} ({df_clean['Num_Alive_Codes'].max():.0f})")
    
    # Analysis by ATOM_L1
    print("\n" + "="*30)
    print("ATOM_L1 ANALYSIS")
    print("="*30)
    
    no_reg = df_clean[df_clean['ATOM_L1'] == 0]
    with_reg = df_clean[df_clean['ATOM_L1'] == 1e-4]
    
    if len(no_reg) > 0 and len(with_reg) > 0:
        print(f"Average R² without regularization: {no_reg['R²'].mean():.4f}")
        print(f"Average R² with regularization: {with_reg['R²'].mean():.4f}")
        print(f"Average L0 without regularization: {no_reg['L0'].mean():.2f}")
        print(f"Average L0 with regularization: {with_reg['L0'].mean():.2f}")
        print(f"Average Alive Codes without regularization: {no_reg['Num_Alive_Codes'].mean():.0f}")
        print(f"Average Alive Codes with regularization: {with_reg['Num_Alive_Codes'].mean():.0f}")
