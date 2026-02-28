import bscope.ic as bic
import pickle
import bscope
import torch
import numpy as np
import os
import glob
import h5py
import bscope.ic as bic
import numpy as np

SEED = 2002
N_SUBSAMPLE = 10
N_TRIALS = 10
PRESERVE = False
LAYERS = [3, 7, 13, 15]  
LAYERS= LAYERS[::-1]
PCTS = [25, 50]
DEVICE = 'cuda:0'

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================================
# PATHS
# ============================================================================
BASE_PATH = '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes'
ALEPH_PATH = os.path.join(BASE_PATH, 'aleph_contributions_positive/mode_summary.h5')
RESULTS_DIR = os.path.expanduser('~/codec_zaki/iclr_scripts/STSAE_ABLATIONS_MODEFIX')
os.makedirs(RESULTS_DIR, exist_ok=True)

hyperparameter_groups = {
    'dict_size': {
        'sweep_dir': 'sweep_int_grad_top_1_False_resnet50_steps_10_483_positive_STSAE',  
        'configs': [
            'hypersweep_0.7_1_0.0001',
            'hypersweep_0.7_3_0.0001',
            'hypersweep_0.7_5_0.0001'
        ]
    },
    
    'threshold': {
        'sweep_dir': 'sweep_int_grad_top_1_False_resnet50_steps_10_483_positive_STSAE', 
        'configs': [
            'hypersweep_0.5_5_0.0001',
            'hypersweep_0.7_5_0.0001',
            'hypersweep_0.9_5_0.0001'
        ]
    },
    
    'mode_l1': {
        'sweep_dir': 'sweep_int_grad_top_1_False_resnet50_steps_10_483_positive_STSAE', 
        'configs': [
            'hypersweep_0.9_5_0',
            'hypersweep_0.9_5_0.01',
            'hypersweep_0.9_5_0.0001'
        ]
    },
    
    'seed': {
        'sweep_dirs': [
            'sweep_int_grad_top_1_False_resnet50_steps_10_1001_positive_STSAE', 
            'sweep_int_grad_top_1_False_resnet50_steps_10_2002_positive_STSAE',  
            'sweep_int_grad_top_1_False_resnet50_steps_10_483_positive_STSAE'  
        ],
        'config': 'hypersweep_0.9_5_0.0001'
    },
    
    'mlp_size': {
        'sweep_dir': 'sweep_int_grad_top_1_False_resnet50_steps_10_483_mlpsize_nonneg_STSAE',
        'configs': [
            'hypersweep_mlpsize_128_nonneg_False_0.5_5_0.0001',
            'hypersweep_mlpsize_512_nonneg_False_0.5_5_0.0001',
            'hypersweep_mlpsize_2048_nonneg_False_0.5_5_0.0001',
            'hypersweep_mlpsize_4096_nonneg_False_0.5_5_0.0001'
        ]
    },
    
    'nonnegative': {
        'sweep_dir': 'sweep_int_grad_top_1_False_resnet50_steps_10_483_mlpsize_nonneg_STSAE', 
        'configs': [
            'hypersweep_mlpsize_4096_nonneg_True_0.5_5_0.0001',
            'hypersweep_mlpsize_2048_nonneg_True_0.5_5_0.0001',
            'hypersweep_mlpsize_512_nonneg_True_0.5_5_0.0001',
        ]
    }
}

all_sweeps = []

for group_name, group_info in hyperparameter_groups.items():
    if group_name == 'seed':
        for sweep_dir in group_info['sweep_dirs']:
            ms_path = os.path.join(BASE_PATH, sweep_dir, group_info['config'], 'mode_summary.h5')
            if os.path.exists(ms_path):
                all_sweeps.append({
                    'path': ms_path,
                    'config_name': group_info['config'],
                    'sweep_name': os.path.basename(sweep_dir),
                    'group': group_name
                })

    else:
        sweep_dir = group_info['sweep_dir']
        for config_name in group_info['configs']:
            ms_path = os.path.join(BASE_PATH, sweep_dir, config_name, 'mode_summary.h5')
            if os.path.exists(ms_path):
                all_sweeps.append({
                    'path': ms_path,
                    'config_name': config_name,
                    'sweep_name': os.path.basename(sweep_dir),
                    'group': group_name
                })

print(f"Found {len(all_sweeps)} sweep configs")
print('USING STSAE')

# LOAD ALEPH
# ============================================================================
print("Loading aleph mode summary...")
aleph_ms = bic.ModeSummary(ALEPH_PATH)

# ============================================================================
# LOAD MODEL
# ============================================================================
print("Loading ResNet50 model...")
model, dataset, dataloader, layers_dict = bscope.ic.get_model(
    imagenet_path='/data/imagenet',
    which_model='resnet50',
    return_layers=True,
    batch_size=128,
    pin_memory=False,
    device=DEVICE,
    shuffle=False)
del dataset, dataloader
model.eval()


# ============================================================================
# RUN COMPARISONS
# ============================================================================
for idx, sweep_info in enumerate(all_sweeps):
    print(f"\n{'='*70}")
    print(f"Processing {idx+1}/{len(all_sweeps)}")
    print(f"Sweep: {sweep_info['sweep_name']}")
    print(f"Config: {sweep_info['config_name']}")
    print(f"{'='*70}")
    
    results_filename = f"{sweep_info['sweep_name']}_{sweep_info['config_name']}_vs_aleph.pkl"
    RESULTS_FILE = os.path.join(RESULTS_DIR, results_filename)
    
    if os.path.exists(RESULTS_FILE):
        print("Already processed, skipping...")
        continue
    
    sweep_ms = bic.ModeSummary(sweep_info['path'])



    # Results structure
    results = {
        'aleph': {li: {pct: {'og_top1': [], 'og_top5': [], 'pert_top1': [], 'pert_top5': [], 'subclasses': [], 'atom_idx': []} for pct in PCTS} for li in LAYERS},
        'sweep': {li: {pct: {'og_top1': [], 'og_top5': [], 'pert_top1': [], 'pert_top5': [], 'subclasses': [], 'atom_idx': []} for pct in PCTS} for li in LAYERS},
        'config_name': sweep_info['config_name'],
        'sweep_name': sweep_info['sweep_name'],
        'group': sweep_info['group'] 
    }
    
    for li in LAYERS:
        print(f"\nLayer {li}")
        
        for trial in range(N_TRIALS):
            subclasses = [np.random.randint(0, 1000), np.random.randint(0, 1000)]
            print(f"  Trial {trial+1}/{N_TRIALS}, Classes: {subclasses}")
            
            dataloader = bic.get_model('resnet50', return_layers=False,
                                       imagenet_path='/data/imagenet', device=DEVICE,
                                       subsample=N_SUBSAMPLE, subclasses=subclasses,
                                       dataloader_only=True, batch_size=50)
            
            # Baseline
            top1, top5 = bic.calculate_subsample_accuracy(model, dataloader, subclasses, device=DEVICE)
            print(f"    Baseline - Top1: {top1}")
            

            aleph_pos = np.where(aleph_ms.layer_idxs == li)[0][0]
            sweep_pos = np.where(sweep_ms.layer_idxs == li)[0][0]

            aleph_idx, aleph_atom, _, aleph_corr = bic.get_top_mode(aleph_ms, aleph_pos, subclasses[0], 0)
            sweep_idx, sweep_atom, _, sweep_corr = bic.get_top_mode(sweep_ms, sweep_pos, subclasses[0], 0)


            
            print(f"    Aleph mode {aleph_idx} (corr={aleph_corr:.3f}), Sweep mode {sweep_idx} (corr={sweep_corr:.3f})")
            
            num_chans = sweep_atom.shape[0]

            
            for pct in PCTS:
                print('----- Percentage:', pct)

                num_to_keep = int(num_chans * pct / 100)
                
                # Aleph ablation
                aleph_channels = list(bic.top_n(aleph_atom, num_to_keep)[0].astype(int))
                pert_channels_aleph = aleph_channels if not PRESERVE else list(set(range(num_chans)) - set(aleph_channels))
                
                disruptor = bscope.Disruptor(layers_dict[li], pert_channels_aleph)
                disruptor.activate()
                aleph_pt1, aleph_pt5 = bic.calculate_subsample_accuracy(model, dataloader, subclasses, device=DEVICE)
                disruptor.deactivate()
                
                # Sweep ablation
                sweep_channels = list(bic.top_n(sweep_atom, num_to_keep)[0].astype(int))
                pert_channels_sweep = sweep_channels if not PRESERVE else list(set(range(num_chans)) - set(sweep_channels))

                disruptor = bscope.Disruptor(layers_dict[li], pert_channels_sweep)
                disruptor.activate()
                sweep_pt1, sweep_pt5 = bic.calculate_subsample_accuracy(model, dataloader, subclasses, device=DEVICE)
                disruptor.deactivate()
                
                print('----- Baseline:', top1, top5)
                print('----- Aleph:', aleph_pt1, aleph_pt5)
                print('----- Sweep:', sweep_pt1, sweep_pt5)
                
                print('----- Aleph drop:', top1[0]-aleph_pt1[0], top5[0]-aleph_pt5[0])
                print('----- Sweep drop:', top1[0]-sweep_pt1[0], top5[0]-sweep_pt5[0])
                
                # Store aleph results
                results['aleph'][li][pct]['og_top1'].append(list(top1))
                results['aleph'][li][pct]['og_top5'].append(list(top5))
                results['aleph'][li][pct]['pert_top1'].append(list(aleph_pt1))
                results['aleph'][li][pct]['pert_top5'].append(list(aleph_pt5))
                results['aleph'][li][pct]['subclasses'].append(subclasses)
                results['aleph'][li][pct]['atom_idx'].append(aleph_idx)
                
                # Store sweep results
                results['sweep'][li][pct]['og_top1'].append(list(top1))
                results['sweep'][li][pct]['og_top5'].append(list(top5))
                results['sweep'][li][pct]['pert_top1'].append(list(sweep_pt1))
                results['sweep'][li][pct]['pert_top5'].append(list(sweep_pt5))
                results['sweep'][li][pct]['subclasses'].append(subclasses)
                results['sweep'][li][pct]['atom_idx'].append(sweep_idx)
    
    # Save
    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved: {RESULTS_FILE}")

print(f"\n{'='*70}")
print("Done!")
print(f"Results in: {RESULTS_DIR}")
print(f"{'='*70}")