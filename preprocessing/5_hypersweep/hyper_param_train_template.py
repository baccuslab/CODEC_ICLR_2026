import torch
from bscope import r2_score
import os
import bscope
from IPython import embed
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import h5py as h5
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import bscope.ic as bic
import wandb

# WandB Sweep Configuration
sweep_config = {
    'method': 'grid',
    'parameters': {
        'N': {
            'values': [1, 3, 8]
        },
        'THRESHOLD': {
            'values': [0.5, 0.7, 0.9]
        },
        'ATOM_L1': {
            'values': [0, 1e-4]  
        }
    }
}

def train_sae():
    # Initialize wandb run
    wandb.init()
    config = wandb.config
    
    # Regularization
    CODE_L1 = None


    # Training parameters
    BATCH_SIZE = 256
    N = config.N  # Dictionary size = N * number of features
    THRESHOLD = config.THRESHOLD
    ATOM_L1 = config.ATOM_L1 if config.ATOM_L1 > 0 else None
    LEARNING_RATE = 5e-4
    EPOCHS = 300
    DEVICE = 'cuda:3'
    wandb.run.name = f"thresh_{THRESHOLD}_{N}_{config.ATOM_L1}"
    # SAE Encoder Parameters


    # Data parameters
    DATA_DIR = '/data/codec/decomps/resnet50_act_normgrad_top_3_nosoftmax/'
    DATA_PATH = os.path.join(DATA_DIR, 'data.h5')
    SAE_PATH = os.path.join(DATA_DIR, 'saes')
    PROJECT_DIR = os.path.join(SAE_PATH, 'sae-sigthresh-sweep')
    if not os.path.exists(PROJECT_DIR):
        os.makedirs(PROJECT_DIR)
    run_dir = os.path.join(PROJECT_DIR, f"hypersweep_{config.THRESHOLD}_{config.N}_{config.ATOM_L1}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    LAYER_IDX = '15'

    DATA_TYPE = 'contributions'  # 'contributions' or 'gradients' or 'activations'
    SPATIAL_OPERATION = 'sum'

    if not os.path.exists(SAE_PATH):
        os.makedirs(SAE_PATH)

    # Load semantic analyzer for correlation analysis
    syn = bic.SemanticAnalyzer('/data/codec/hierarchy_metadata/semantic_indexes.json')
    # Load data
    data, targets = bic.load_contribution_data(DATA_PATH, DATA_TYPE, LAYER_IDX, SPATIAL_OPERATION, norm=True)
    dataset = TensorDataset(torch.from_numpy(data).float())
    HIDDEN_SIZE = 2048

    # Get semantic masks for correlation analysis
    mask_mtx, mask_labels = syn.get_all_semantic_masks(targets)

    # Initialize SAE
    sae = bscope.SigThreshSAE(data.shape[-1],
            num_atoms=int(data.shape[-1]*N),
            threshold=THRESHOLD,
            mlp_hidden_dim=HIDDEN_SIZE).to(DEVICE)

    # Data loaders
    train_dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True,
                                             pin_memory=False,
                                             num_workers=3)

    eval_dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False,
                                             pin_memory=False,
                                             num_workers=3)

    optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)

    # Training loop
    for epoch in range(EPOCHS):
        losses = [] 
        r2_scores = []

        sae.train()
        for train_batch in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            train_batch = train_batch[0].to(DEVICE)
            optimizer.zero_grad()

            pre_codes, codes, reconstructed = sae(train_batch)

            # Calculate R2 score
            r2 = r2_score(train_batch, reconstructed)
            r2_scores.append(r2.detach().cpu().numpy())

            loss = loss_fn(reconstructed, train_batch)

            if CODE_L1 is not None:
                loss += CODE_L1 * torch.mean(torch.abs(codes))
            if ATOM_L1 is not None:
                loss += ATOM_L1 * torch.mean(torch.abs(sae.dictionary.get_dictionary()))

            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            optimizer.step()

        # Calculate epoch metrics
        epoch_loss = np.mean(losses)
        epoch_r2 = np.mean(r2_scores)
        learning_rate = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, R2: {epoch_r2:.4f}, LR: {learning_rate:.6f}')
        
        # Log basic metrics every epoch
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_r2': epoch_r2,
            'learning_rate': learning_rate
        })

        scheduler.step()

        # Save model every 20 epochs
        if epoch % 30 == 0:
                    model_path = os.path.join(run_dir, f'sae_epoch_{epoch+1}.pt')
                    torch.save(sae, model_path)
                    wandb.log({'checkpoint_path': model_path})

        # Evaluation and correlation analysis every 5 epochs
        if epoch % 10 == 0:
            codes_agg = []
            z_agg = []
            eval_r2_scores = []
            
            sae.eval()
            with torch.no_grad():
                for eval_batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
                    eval_batch = eval_batch[0].to(DEVICE)
                    
                    pre_codes, codes, reconstructed = sae(eval_batch)
                    z_agg.append(pre_codes.detach().cpu().numpy())
                    codes_agg.append(codes.detach().cpu().numpy())
                    
                    r2 = r2_score(eval_batch, reconstructed)
                    eval_r2_scores.append(r2.detach().cpu().numpy())

            # Aggregate results
            codes_agg = np.concatenate(codes_agg, axis=0) 
            z_agg = np.concatenate(z_agg, axis=0)
            eval_r2 = np.mean(eval_r2_scores)
            firings=np.sum(codes_agg, 0)
            alive=firings>0
            alive_codes=codes_agg[:,alive]
            dictionary=sae.dictionary.get_dictionary().detach().cpu().numpy()
            print(f'Alive Codes: {alive.sum()}')

            # Correlation analysis
            corr_mtx = bscope.mtx_corr(alive_codes, mask_mtx.T)
            corr_mtx[np.isnan(corr_mtx)] = 0

            # Calculate correlation metrics
            max_corr = np.max(corr_mtx)
            mean_corr = np.mean(corr_mtx)
            
            # Top 10 correlations
            flat_corr = corr_mtx.flatten()
            top_indices = np.argsort(flat_corr)[-10:][::-1]
            top_10_corrs = []
            
            for i, flat_idx in enumerate(top_indices):
                feature_idx, concept_idx = np.unravel_index(flat_idx, corr_mtx.shape)
                corr_val = corr_mtx[feature_idx, concept_idx]
                top_10_corrs.append(corr_val)
            
            avg_top10_corr = np.mean(top_10_corrs)
            
            # Count positive correlations
            positive_corrs = np.sum(corr_mtx > 0.2)
            features_with_pos_corrs = np.sum((corr_mtx > 0.2).sum(0) > 0)

            print(f'===== Evaluation Results (Epoch {epoch+1}) =====')
            print(f'Eval R2: {eval_r2:.4f}')
            print(f'Alive modes: {alive.sum()}, Dead modes: {np.sum(codes_agg==0)}')
            print(f'Max correlation: {max_corr:.4f}')
            print(f'Average top 10 correlations: {avg_top10_corr:.4f}')
            print(f'Positive correlations (>0.2): {positive_corrs}')
            print('=' * 50)

            # Log evaluation metrics
            wandb.log({
                'eval_r2': eval_r2,
                'alive_modes': alive.sum(),
                'dead_modes': {np.sum(codes_agg==0)},
                'max_correlation': max_corr,
                'mean_correlation': mean_corr,
                'avg_top10_corr': avg_top10_corr,
                'positive_correlations': positive_corrs,
                'features_with_pos_corrs': features_with_pos_corrs,
                'top_1_corr': top_10_corrs[0] if len(top_10_corrs) > 0 else 0,
                'top_5_avg_corr': np.mean(top_10_corrs[:5]) if len(top_10_corrs) >= 5 else 0
            })


    
    # Also save the full model
    full_model_path = os.path.join(run_dir, f'sae_final_model_epoch_{EPOCHS}.pt')
    torch.save(sae, full_model_path)

    # Final evaluation
    with torch.no_grad():
        codes_agg = []
        z_agg = []
        eval_r2_scores = []
            
        sae.eval()
        with torch.no_grad():
            for eval_batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
                eval_batch = eval_batch[0].to(DEVICE)
                
                pre_codes, codes, reconstructed = sae(eval_batch)
                z_agg.append(pre_codes.detach().cpu().numpy())
                codes_agg.append(codes.detach().cpu().numpy())
                
                r2 = r2_score(eval_batch, reconstructed)
                eval_r2_scores.append(r2.detach().cpu().numpy())

        # Aggregate results
        codes_agg = np.concatenate(codes_agg, axis=0) 
        z_agg = np.concatenate(z_agg, axis=0)
        eval_r2 = np.mean(eval_r2_scores)
        firings=np.sum(codes_agg, 0)
        alive=firings>0
        alive_codes=codes_agg[:,alive]
        dictionary=sae.dictionary.get_dictionary().detach().cpu().numpy()
        print(f'Alive Codes: {dictionary.shape[0]}')

        # Correlation analysis
        corr_mtx = bscope.mtx_corr(alive_codes, mask_mtx.T)
        corr_mtx[np.isnan(corr_mtx)] = 0

        # Calculate correlation metrics
        max_corr = np.max(corr_mtx)
        mean_corr = np.mean(corr_mtx)
        
        # Top 10 correlations
        flat_corr = corr_mtx.flatten()
        top_indices = np.argsort(flat_corr)[-10:][::-1]
        top_10_corrs = []
        
        for i, flat_idx in enumerate(top_indices):
            feature_idx, concept_idx = np.unravel_index(flat_idx, corr_mtx.shape)
            corr_val = corr_mtx[feature_idx, concept_idx]
            top_10_corrs.append(corr_val)
        
        avg_top10_corr = np.mean(top_10_corrs)
        
        # Count positive correlations
        positive_corrs = np.sum(corr_mtx > 0.2)
        features_with_pos_corrs = np.sum((corr_mtx > 0.2).sum(0) > 0)

        print(f'===== Evaluation Results (Epoch {epoch+1}) =====')
        print(f'Eval R2: {eval_r2:.4f}')
        print(f'Alive modes: {np.sum(firings>0)}, Dead modes: {np.sum(firings==0)}')
        print(f'Max correlation: {max_corr:.4f}')
        print(f'Average top 10 correlations: {avg_top10_corr:.4f}')
        print(f'Positive correlations (>0.2): {positive_corrs}')
        print('=' * 50)

        # Log evaluation metrics
        wandb.log({
            'eval_r2': eval_r2,
            'alive_modes': np.sum(firings>0),
            'dead_modes': np.sum(firings==0),
            'max_correlation': max_corr,
            'mean_correlation': mean_corr,
            'avg_top10_corr': avg_top10_corr,
            'positive_correlations': positive_corrs,
            'features_with_pos_corrs': features_with_pos_corrs,
            'top_1_corr': top_10_corrs[0] if len(top_10_corrs) > 0 else 0,
            'top_5_avg_corr': np.mean(top_10_corrs[:5]) if len(top_10_corrs) >= 5 else 0
        })

    
    wandb.log({'model_path': model_path})
    wandb.finish()

# Initialize and run sweep
if __name__ == "__main__":
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project="sae-sigthresh-sweep")
    
    # Run sweep
    wandb.agent(sweep_id, train_sae)  # 25 = 5x5 grid