import bscope.ic as bic
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import glob
import os
import bscope
from fycus import Fycus

F = Fycus('hyperparam', base_path='/home/zalaoui/higanbana/STSAE')


# Black widow spider class index
CLASS_IDX = 75
LAYER = 15
CORR_THRESHOLD = 0.2

mode_types = ['sum']



for mode in mode_types:
    mode_type = mode

    # Paths - all three seeds
    ALEPH_PATH = '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/aleph_contributions_positive/mode_summary.h5'
    SWEEP_BASE_1001 = '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/sweep_int_grad_top_1_False_resnet50_steps_10_1001_positive_STSAE'
    SWEEP_BASE_2002 = '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/sweep_int_grad_top_1_False_resnet50_steps_10_2002_positive_STSAE'
    SWEEP_BASE_483 = '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/sweep_int_grad_top_1_False_resnet50_steps_10_483_positive_STSAE'
    SWEEP_BASE_MLP = '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/sweep_int_grad_top_1_False_resnet50_steps_10_483_mlpsize_nonneg_STSAE'



    # Collect all configs with seed info
    parsed_configs = []

    for seed, sweep_base in [(1001, SWEEP_BASE_1001), (2002, SWEEP_BASE_2002), (483, SWEEP_BASE_483)]:
        for cfg_path in glob.glob(os.path.join(sweep_base, 'hypersweep_*')):
            cfg_name = os.path.basename(cfg_path)
            params = bscope.parse_config(cfg_name)
            if params:
                params['path'] = os.path.join(cfg_path, 'mode_summary.h5')
                params['name'] = cfg_name
                params['seed'] = seed
                parsed_configs.append(params)

    for cfg_path in glob.glob(os.path.join(SWEEP_BASE_MLP, 'hypersweep_*')):
        cfg_name = os.path.basename(cfg_path)
        params = bscope.parse_config(cfg_name)
        if params:
            params['path'] = os.path.join(cfg_path, 'mode_summary.h5')
            params['name'] = cfg_name
            params['seed'] = 483
            parsed_configs.append(params)

    print(f"Found {len(parsed_configs)} configs")

    # Get baseline
    result = bic.get_summed_atom(ALEPH_PATH, LAYER, CLASS_IDX, mode=mode_type)
    baseline_atom = result[0] if result else None
    if baseline_atom is not None:
        baseline_atom = baseline_atom / np.linalg.norm(baseline_atom)

    # Organize configs
    grid_configs = [c for c in parsed_configs if c['sweep_type'] == 'grid']
    mlp_configs = [c for c in parsed_configs if c['sweep_type'] == 'mlp']

    # Get unique values for each parameter
    thresholds = sorted(list(set(c['threshold'] for c in grid_configs)))
    Ns = sorted(list(set(c['N'] for c in grid_configs)))
    atom_l1s = sorted(list(set(c['atom_l1'] for c in grid_configs)))
    seeds = sorted(list(set(c['seed'] for c in grid_configs)))



    # MLP sweeps
    mlp_configs_true = [c for c in mlp_configs if c['nonneg'] == False]
    mlp_sizes = sorted(list(set(c['mlp_size'] for c in mlp_configs_true)))
    mlp_sizes_all = sorted(list(set(c['mlp_size'] for c in mlp_configs)))

    # Fixed values
    fixed_N = Ns[len(Ns)//2]
    fixed_l1 = atom_l1s[len(atom_l1s)//2]
    fixed_seed = 1001
    fixed_thresh = thresholds[len(thresholds)//2]
    fixed_mlp = mlp_sizes_all[len(mlp_sizes_all)//2]

    # Define parameter sweeps
    param_sweeps = [
        ('atom_l1', atom_l1s, 'Atom L1', 
         [f'{l:.0e}' if l != 0 else '0' for l in atom_l1s],
         lambda l: next((c for c in grid_configs if c['threshold'] == fixed_thresh and c['N'] == fixed_N and c['atom_l1'] == l and c['seed'] == fixed_seed), None)),
        
        ('threshold', thresholds, 'Loading thresh.', 
         [f'{t}' for t in thresholds],
         lambda t: next((c for c in grid_configs if c['threshold'] == t and c['N'] == fixed_N and c['atom_l1'] == fixed_l1 and c['seed'] == fixed_seed), None)),
        
        ('N', Ns, 'Num. atoms', 
         [f'{n}' for n in Ns],
         lambda n: next((c for c in grid_configs if c['threshold'] == fixed_thresh and c['N'] == n and c['atom_l1'] == fixed_l1 and c['seed'] == fixed_seed), None)),
        
        ('seed', seeds, 'Seed', 
         [f'{s}' for s in seeds],
         lambda s: next((c for c in grid_configs if c['threshold'] == fixed_thresh and c['N'] == fixed_N and c['atom_l1'] == fixed_l1 and c['seed'] == s), None)),
        
        ('mlp_size', mlp_sizes, 'MLP size', 
         [f'{s}' for s in mlp_sizes],
         lambda s: next((c for c in mlp_configs_true if c['mlp_size'] == s), None)),
        
        ('nonneg', [True, False], 'Atom sign', 
        ['Positive constraint', 'No constraint'],  # Fixed: swapped the order
        lambda n: next((c for c in mlp_configs if c['mlp_size'] == fixed_mlp and c['nonneg'] == n), None)),
    ]


    max_n_values = max(len(param_values) for _, param_values, _, _, _ in param_sweeps)
    

    n_cols = 6
    n_rows = max_n_values + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 2.5 * n_rows))
    
    for col_idx, (param_name, param_values, param_title, value_labels, get_config) in enumerate(param_sweeps):
        n_values = len(param_values)
        
        # Row 0: Baseline
        ax = axes[0, col_idx]
        if baseline_atom is not None:
            ax.plot(baseline_atom, 'k-', linewidth=1.5)
        ax.set_title(param_title, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel('Baseline')
        ax.set_xticklabels([])
        
        # Rows 1+: Each parameter value
        for i, param_val in enumerate(param_values):
            ax = axes[i + 1, col_idx]
            
            cfg = get_config(param_val)
            if cfg:
                result = bic.get_summed_atom(cfg['path'], LAYER, CLASS_IDX, mode=mode_type)
                if result:
                    atom = result[0] / np.linalg.norm(result[0])
                    ax.plot(atom, 'k-', linewidth=1.5)
            
       
            ax.set_ylabel(value_labels[i])
            
            # Only show x-label on bottom row
            if i == n_values - 1:
                ax.set_xlabel('Channel')
            else:
                ax.set_xticklabels([])
            

        
        # Hide unused rows for this column
        for i in range(n_values, max_n_values):
            axes[i + 1, col_idx].axis('off')
    
    plt.tight_layout()
    # F.XX(1.0, 1.5)
    # F.save(f'single_mode_vector_comparison_class{CLASS_IDX}_layer{LAYER}_all_params_{mode_type}')
    plt.show()
