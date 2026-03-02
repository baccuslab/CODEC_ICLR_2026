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

LAYERS_TO_TRAIN = [3, 7, 13, 15]  
sweep_config = {
    'method': 'grid',
    'parameters': {
        'NONNEGATIVE': {
            'values': [True, False]
        },
        'MLP_SIZE': {
            'values': [128,512,2048,4096]
        },

    }
}

def train_sae(seed):
    # Initialize wandb run
    wandb.init()
    config = wandb.config
    




    # Training parameters
    BATCH_SIZE = 128
    CODE_L1 = None
    N = 5 
    THRESHOLD = .5
    MODE_L1 = 1e-4
    EPOCHS = 200
    LEARNING_RATE = 5e-5
    MLP_HIDDEN_SIZE = config.MLP_SIZE
    NONNEGATIVE_DICT = config.NONNEGATIVE
    DEVICE = 'cuda:1'
    print(f'\n{"="*50}')
    print(f'Running with seed: {seed}')
    print(f'{"="*50}\n')
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    wandb.run.name = f"thresh_{THRESHOLD}_{N}_{MODE_L1}_mlpsize_{MLP_HIDDEN_SIZE}_nonneg_{NONNEGATIVE_DICT}"



    # Data parameters
    DATA_DIR = '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10'
    DATA_PATH = os.path.join(DATA_DIR, 'data.h5')
    PROJECT_DIR = os.path.join(DATA_DIR, 'saes', project_name)
    if not os.path.exists(PROJECT_DIR):
        os.makedirs(PROJECT_DIR)
    run_dir = os.path.join(PROJECT_DIR, f"hypersweep_mlpsize_{MLP_HIDDEN_SIZE}_nonneg_{NONNEGATIVE_DICT}_{THRESHOLD}_{N}_{MODE_L1}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    print(f"Run directory: {run_dir}")
    print(f'Project directory: {PROJECT_DIR}')


    DATA_TYPE = 'contributions'  # 'contributions' or 'gradients' or 'activations'
    SPATIAL_OPERATION = 'positive'



    for LAYER_IDX in LAYERS_TO_TRAIN:
        print(f'\n{"="*50}')
        print(f'Training Layer {LAYER_IDX}')

        # Load data
        data, targets = bic.load_contribution_data(DATA_PATH, DATA_TYPE, LAYER_IDX, SPATIAL_OPERATION, norm=True)
        dataset = TensorDataset(torch.from_numpy(data).float())



        # Initialize SAE
        # sae = bscope.SigThreshSAE(data.shape[-1],
        #         num_atoms=int(data.shape[-1]*N),
        #         threshold=THRESHOLD,
        #         mlp_hidden_dim=MLP_HIDDEN_SIZE, nonnegative=NONNEGATIVE_DICT).to(DEVICE)
        
        sae = bscope.STSAE(data.shape[-1],
                num_atoms=data.shape[-1]*N,
                threshold=THRESHOLD,
                mlp_hidden_dim=MLP_HIDDEN_SIZE, nonnegative=NONNEGATIVE_DICT).to(DEVICE)

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
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)

        # Training loop
        for epoch in range(EPOCHS):
            loss_agg = 0 
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
                if MODE_L1 is not None:
                    loss += MODE_L1 * torch.mean(torch.abs(sae.dictionary.get_dictionary()))

                loss.backward()
                loss_agg += loss.item()
                optimizer.step()

            # Calculate epoch metrics
            losses = loss_agg / len(train_dataloader)
            epoch_loss = losses
            epoch_r2 = np.mean(r2_scores)
            learning_rate = optimizer.param_groups[0]['lr']

            print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {losses:.4f}, R2: {epoch_r2:.4f}, LR: {learning_rate:.6f}')
            
            wandb.log({
                f'layer_{LAYER_IDX}_epoch': epoch + 1,
                f'layer_{LAYER_IDX}_train_loss': epoch_loss,
                f'layer_{LAYER_IDX}_train_r2': epoch_r2,
                f'layer_{LAYER_IDX}_learning_rate': learning_rate
            })

            # scheduler.step()

            if epoch % 100 == 0:

                        model_path = os.path.join(run_dir, f'sae_layer_{LAYER_IDX}_epoch_{epoch+1}.pt')
                        torch.save(sae, model_path)
                        wandb.log({'checkpoint_path': model_path})

            if epoch % 10 == 0:
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
                        



                codes_agg = np.concatenate(codes_agg, axis=0) 
                eval_data_agg = np.concatenate(eval_data_agg, axis=0)
                eval_reconstructed_agg = np.concatenate(eval_reconstructed_agg, axis=0)

                eval_data_agg = torch.from_numpy(eval_data_agg).float()
                eval_reconstructed_agg = torch.from_numpy(eval_reconstructed_agg).float()

                r2 = r2_score(eval_data_agg, eval_reconstructed_agg)

                firings=np.sum(codes_agg, 0)
                alive=firings>0
                alive_codes=codes_agg[:,alive]
                dictionary=sae.dictionary.get_dictionary().detach().cpu().numpy()
                print(f'Alive Codes: {dictionary.shape[0]}')



                wandb.log({
                    f'layer_{LAYER_IDX}_eval_r2': r2,
                    f'layer_{LAYER_IDX}_alive_modes': np.sum(firings>0),
                    f'layer_{LAYER_IDX}_dead_modes': np.sum(firings==0),
                })


        

        model_path = os.path.join(run_dir, f'sae_layer_{LAYER_IDX}_epoch_{epoch+1}.pt')
        full_model_path = os.path.join(run_dir, f'sae_layer_{LAYER_IDX}_final_model.pt')
        torch.save(sae, full_model_path)

        # Final evaluation
        with torch.no_grad():
            codes_agg = []
            eval_data_agg=[]
            eval_reconstructed_agg=[]
                
            sae.eval()
            with torch.no_grad():
                for eval_batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
                    eval_batch = eval_batch[0].to(DEVICE)
                    
                    pre_codes, codes, reconstructed = sae(eval_batch)
                    codes_agg.append(codes.detach().cpu().numpy())
                    eval_data_agg.append(eval_batch.cpu().detach().numpy())
                    eval_reconstructed_agg.append(reconstructed.detach().cpu().numpy())
                    



            codes_agg = np.concatenate(codes_agg, axis=0) 
            eval_data_agg = np.concatenate(eval_data_agg, axis=0)
            eval_reconstructed_agg = np.concatenate(eval_reconstructed_agg, axis=0)

            eval_data_agg = torch.from_numpy(eval_data_agg).float()
            eval_reconstructed_agg = torch.from_numpy(eval_reconstructed_agg).float()

            r2 = r2_score(eval_data_agg, eval_reconstructed_agg)
            print(f'     Epoch {epoch+1}/{EPOCHS}, R2 Score: {r2:.4f}')

            firings=np.sum(codes_agg, 0)
            alive=firings>0
            alive_codes=codes_agg[:,alive]
            dictionary=sae.dictionary.get_dictionary().detach().cpu().numpy()
            print(f'Alive Codes: {dictionary.shape[0]}')



            print(f'===== Evaluation Results (Epoch {epoch+1}) =====')
            print(f'Eval R2: {r2:.4f}')
            print(f'Alive modes: {np.sum(firings>0)}, Dead modes: {np.sum(firings==0)}')


            print('=' * 50)

            # Log evaluation metrics
            wandb.log({
                f'layer_{LAYER_IDX}_final_eval_r2': r2,
                f'layer_{LAYER_IDX}_final_alive_modes': np.sum(firings>0),
                f'layer_{LAYER_IDX}_final_dead_modes': np.sum(firings==0),
            })

        
        wandb.log({'model_path': model_path})
    wandb.finish()

# # Initialize and run sweep
num_configs = 1
for param in sweep_config['parameters'].values():
    num_configs *= len(param['values'])
print(num_configs)
if __name__ == "__main__":
    SEEDS = [483]
    
    for seed in SEEDS:

        project_name = f'sweep_int_grad_top_1_False_resnet50_steps_10_{seed}_mlpsize_nonneg_STSAE'
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        wandb.agent(sweep_id, lambda s=seed: train_sae(seed=s), count=num_configs)