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
import random
import time
from collections import defaultdict

LAYERS_TO_TRAIN = [3,7,13,15]  # Define which layers

SEED = 1001
# Training parameters
BATCH_SIZE = 128
N = 5  
THRESHOLD = .9
MODE_L1 = 1e-4
EPOCHS = 200
LEARNING_RATE = 5e-5
project_name = 'run_time_metrics'

# Timing and memory tracking
timing_stats = defaultdict(list)
overall_start_time = time.time()

# Initialize wandb run
wandb.init(project=project_name, config={
    'N': N,
    'THRESHOLD': THRESHOLD,
    'MODE_L1': MODE_L1,
    'seed': SEED,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'epochs': EPOCHS
})

# Regularization
CODE_L1 = None
MLP_HIDDEN_SIZE = None
DEVICE = 'cuda:2'

print(f'\n{"="*50}')
print(f'Running with seed: {SEED}')
print(f'N: {N}, THRESHOLD: {THRESHOLD}, ATOM_L1: {MODE_L1}')
print(f'{"="*50}\n')

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

wandb.run.name = f"single_run_thresh_{THRESHOLD}_N_{N}_ATOM_{ATOM_L1}"

# Data parameters
DATA_DIR = '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/'
DATA_PATH = os.path.join(DATA_DIR, 'data.h5')
PROJECT_DIR = os.path.join(DATA_DIR, 'saes')
if not os.path.exists(PROJECT_DIR):
    os.makedirs(PROJECT_DIR)

run_dir = os.path.join(PROJECT_DIR, f"{THRESHOLD}_{N}_{ATOM_L1}_layer3")
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

print(f"Run directory: {run_dir}")
print(f'Project directory: {PROJECT_DIR}')

DATA_TYPE = 'contributions'
SPATIAL_OPERATION = 'positive'

# Train on each layer
for LAYER_IDX in LAYERS_TO_TRAIN:
    layer_start_time = time.time()
    
    print(f'\n{"="*50}')
    print(f'Training Layer {LAYER_IDX}')
    
    # Load data - TIME THIS
    data_load_start = time.time()
    data, targets = bic.load_contribution_data(DATA_PATH, DATA_TYPE, LAYER_IDX, SPATIAL_OPERATION, norm=True)
    data_load_time = time.time() - data_load_start
    timing_stats[f'layer_{LAYER_IDX}_data_loading'].append(data_load_time)
    print(f"Data loading time: {data_load_time:.2f}s")
    
    if MLP_HIDDEN_SIZE is None: 
        MLP_HIDDEN_SIZE = data.shape[-1]
    
    dataset = TensorDataset(torch.from_numpy(data).float())
    
    # Initialize SAE - TIME THIS
    model_init_start = time.time()
    sae = bscope.STSAE(data.shape[-1],
            num_atoms=int(data.shape[-1]*N),
            threshold=THRESHOLD,
            mlp_hidden_dim=MLP_HIDDEN_SIZE).to(DEVICE)
    model_init_time = time.time() - model_init_start
    timing_stats[f'layer_{LAYER_IDX}_model_init'].append(model_init_time)
    
    # Count model parameters
    num_params = sum(p.numel() for p in sae.parameters())
    num_trainable_params = sum(p.numel() for p in sae.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} (trainable: {num_trainable_params:,})")
    print(f"Model initialization time: {model_init_time:.2f}s")
    
    # Get initial memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
        initial_memory = torch.cuda.memory_allocated(DEVICE) / 1e9
        print(f"Initial GPU memory: {initial_memory:.2f} GB")
    
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
    
    # Training loop
    epoch_times = []
    forward_times = []
    backward_times = []
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        loss_agg = 0 
        r2_scores = []
        
        batch_forward_times = []
        batch_backward_times = []
        
        sae.train()
        for train_batch in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            train_batch = train_batch[0].to(DEVICE)
            
            forward_start = time.time()
            optimizer.zero_grad()
            pre_codes, codes, reconstructed = sae(train_batch)
            

            r2 = r2_score(train_batch, reconstructed)
            r2_scores.append(r2.detach().cpu().numpy())
            
            loss = loss_fn(reconstructed, train_batch)
            if CODE_L1 is not None:
                loss += CODE_L1 * torch.mean(torch.abs(codes))
            if ATOM_L1 is not None:
                loss += ATOM_L1 * torch.mean(torch.abs(sae.dictionary.get_dictionary()))
            
            forward_time = time.time() - forward_start
            batch_forward_times.append(forward_time)
            

            backward_start = time.time()
            loss.backward()
            loss_agg += loss.item()
            optimizer.step()
            backward_time = time.time() - backward_start
            batch_backward_times.append(backward_time)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_forward_time = np.mean(batch_forward_times)
        avg_backward_time = np.mean(batch_backward_times)
        forward_times.append(avg_forward_time)
        backward_times.append(avg_backward_time)
        
        # Calculate epoch metrics
        losses = loss_agg / len(train_dataloader)
        epoch_loss = losses
        epoch_r2 = np.mean(r2_scores)
        learning_rate = optimizer.param_groups[0]['lr']
        
        # Get memory stats
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated(DEVICE) / 1e9
            peak_memory = torch.cuda.max_memory_allocated(DEVICE) / 1e9
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {losses:.4f}, R2: {epoch_r2:.4f}, '
              f'Time: {epoch_time:.2f}s (fwd: {avg_forward_time*1000:.1f}ms, bwd: {avg_backward_time*1000:.1f}ms), '
              f'LR: {learning_rate:.6f}')
        
        if torch.cuda.is_available():
            print(f'  Memory - Current: {current_memory:.2f}GB, Peak: {peak_memory:.2f}GB')
        
        # Log basic metrics every epoch
        log_dict = {
            f'layer_{LAYER_IDX}_epoch': epoch + 1,
            f'layer_{LAYER_IDX}_train_loss': epoch_loss,
            f'layer_{LAYER_IDX}_train_r2': epoch_r2,
            f'layer_{LAYER_IDX}_learning_rate': learning_rate,
            f'layer_{LAYER_IDX}_epoch_time': epoch_time,
            f'layer_{LAYER_IDX}_avg_forward_time_ms': avg_forward_time * 1000,
            f'layer_{LAYER_IDX}_avg_backward_time_ms': avg_backward_time * 1000,
        }
        
        if torch.cuda.is_available():
            log_dict.update({
                f'layer_{LAYER_IDX}_current_memory_gb': current_memory,
                f'layer_{LAYER_IDX}_peak_memory_gb': peak_memory,
            })
        
        wandb.log(log_dict)
        
        # Save model every 100 epochs
        if epoch % 100 == 0:
            save_start = time.time()
            model_path = os.path.join(run_dir, f'sae_layer_{LAYER_IDX}_epoch_{epoch+1}.pt')
            torch.save(sae, model_path)
            save_time = time.time() - save_start
            wandb.log({
                'checkpoint_path': model_path,
                f'layer_{LAYER_IDX}_save_time': save_time
            })
        
        # Evaluation every 10 epochs
        if epoch % 10 == 0:
            eval_start = time.time()
            eval_data_agg = []
            eval_reconstructed_agg = []
            codes_agg = []
            
            sae.eval()
            with torch.no_grad():
                for eval_batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
                    eval_batch = eval_batch[0].to(DEVICE)
                    
                    pre_codes, codes, reconstructed = sae(eval_batch)
                    codes_agg.append(codes.detach().cpu().numpy())
                    eval_data_agg.append(eval_batch.cpu().detach().numpy())
                    eval_reconstructed_agg.append(reconstructed.detach().cpu().numpy())
            
            eval_time = time.time() - eval_start
            
            # Aggregate results
            codes_agg = np.concatenate(codes_agg, axis=0) 
            eval_data_agg = np.concatenate(eval_data_agg, axis=0)
            eval_reconstructed_agg = np.concatenate(eval_reconstructed_agg, axis=0)
            eval_data_agg = torch.from_numpy(eval_data_agg).float()
            eval_reconstructed_agg = torch.from_numpy(eval_reconstructed_agg).float()
            r2 = r2_score(eval_data_agg, eval_reconstructed_agg)
            firings = np.sum(codes_agg, 0)
            alive = firings > 0
            alive_codes = codes_agg[:, alive]
            dictionary = sae.dictionary.get_dictionary().detach().cpu().numpy()
            print(f'Alive Codes: {np.sum(alive)}, Eval time: {eval_time:.2f}s')
            
            # Log evaluation metrics
            wandb.log({
                f'layer_{LAYER_IDX}_eval_r2': r2,
                f'layer_{LAYER_IDX}_alive_modes': np.sum(firings>0),
                f'layer_{LAYER_IDX}_dead_modes': np.sum(firings==0),
                f'layer_{LAYER_IDX}_eval_time': eval_time,
            })
    
    # Save final model
    model_path = os.path.join(run_dir, f'sae_layer_{LAYER_IDX}_epoch_{epoch+1}.pt')
    full_model_path = os.path.join(run_dir, f'sae_layer_{LAYER_IDX}_final_model.pt')
    torch.save(sae, full_model_path)
    
    # Final evaluation
    final_eval_start = time.time()
    with torch.no_grad():
        codes_agg = []
        eval_data_agg = []
        eval_reconstructed_agg = []
        
        sae.eval()
        with torch.no_grad():
            for eval_batch in tqdm.tqdm(eval_dataloader, desc="Final Evaluation"):
                eval_batch = eval_batch[0].to(DEVICE)
                
                pre_codes, codes, reconstructed = sae(eval_batch)
                codes_agg.append(codes.detach().cpu().numpy())
                eval_data_agg.append(eval_batch.cpu().detach().numpy())
                eval_reconstructed_agg.append(reconstructed.detach().cpu().numpy())
        
        # Aggregate results
        codes_agg = np.concatenate(codes_agg, axis=0) 
        eval_data_agg = np.concatenate(eval_data_agg, axis=0)
        eval_reconstructed_agg = np.concatenate(eval_reconstructed_agg, axis=0)
        eval_data_agg = torch.from_numpy(eval_data_agg).float()
        eval_reconstructed_agg = torch.from_numpy(eval_reconstructed_agg).float()
        r2 = r2_score(eval_data_agg, eval_reconstructed_agg)
        final_eval_time = time.time() - final_eval_start
        
        print(f'     Final Epoch {epoch+1}/{EPOCHS}, R2 Score: {r2:.4f}')
        firings = np.sum(codes_agg, 0)
        alive = firings > 0
        alive_codes = codes_agg[:, alive]
        dictionary = sae.dictionary.get_dictionary().detach().cpu().numpy()
        print(f'Alive Codes: {np.sum(alive)}')
        print(f'===== Final Evaluation Results =====')
        print(f'Eval R2: {r2:.4f}')
        print(f'Alive modes: {np.sum(firings>0)}, Dead modes: {np.sum(firings==0)}')
        print('=' * 50)
        
        # Log final metrics
        wandb.log({
            f'layer_{LAYER_IDX}_final_eval_r2': r2,
            f'layer_{LAYER_IDX}_final_alive_modes': np.sum(firings>0),
            f'layer_{LAYER_IDX}_final_dead_modes': np.sum(firings==0),
            f'layer_{LAYER_IDX}_final_eval_time': final_eval_time,
        })
    
    layer_total_time = time.time() - layer_start_time
    
    # Log layer-level timing statistics
    print(f'\n{"="*50}')
    print(f'Layer {LAYER_IDX} Timing Summary:')
    print(f'  Total time: {layer_total_time:.2f}s ({layer_total_time/60:.2f} min)')
    print(f'  Data loading: {data_load_time:.2f}s')
    print(f'  Model init: {model_init_time:.2f}s')
    print(f'  Average epoch time: {np.mean(epoch_times):.2f}s')
    print(f'  Average forward pass: {np.mean(forward_times)*1000:.2f}ms')
    print(f'  Average backward pass: {np.mean(backward_times)*1000:.2f}ms')
    print(f'  Final eval time: {final_eval_time:.2f}s')
    print(f'{"="*50}\n')
    
    wandb.log({
        f'layer_{LAYER_IDX}_total_time': layer_total_time,
        f'layer_{LAYER_IDX}_total_time_minutes': layer_total_time / 60,
        f'layer_{LAYER_IDX}_avg_epoch_time': np.mean(epoch_times),
        f'layer_{LAYER_IDX}_avg_forward_time_ms': np.mean(forward_times) * 1000,
        f'layer_{LAYER_IDX}_avg_backward_time_ms': np.mean(backward_times) * 1000,
        f'layer_{LAYER_IDX}_num_parameters': num_params,
        f'layer_{LAYER_IDX}_num_trainable_parameters': num_trainable_params,
        'model_path': model_path
    })

# Overall timing summary
total_time = time.time() - overall_start_time
print(f'\n{"="*50}')
print(f'OVERALL TIMING SUMMARY')
print(f'{"="*50}')
print(f'Total runtime: {total_time:.2f}s ({total_time/60:.2f} min, {total_time/3600:.2f} hours)')
print(f'Layers trained: {LAYERS_TO_TRAIN}')
print(f'Average time per layer: {total_time/len(LAYERS_TO_TRAIN):.2f}s')
print(f'{"="*50}\n')

wandb.log({
    'total_runtime_seconds': total_time,
    'total_runtime_minutes': total_time / 60,
    'total_runtime_hours': total_time / 3600,
    'avg_time_per_layer_seconds': total_time / len(LAYERS_TO_TRAIN),
})

wandb.finish()
print("\nTraining complete!")