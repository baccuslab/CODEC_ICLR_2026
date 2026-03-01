import bscope.ic as bic
import bscope
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from fycus import Fycus

F = Fycus(fig_dir='hyperparam', base_path='/home/zalaoui/higanbana/STSAE')

# ============================================================================
# CONFIGURATION
# ============================================================================
ALEPH_CONTRIB_PATH = '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/aleph_contributions_positive/mode_summary.h5'
ALEPH_ACTIV_PATH = '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/aleph_activations_positive/mode_summary.h5'

CONCEPT_NAME = 'dog.n.01'  # The concept
LAYERS = [12,13, 14,15]  # List of layers to ablate
CORR_THRESHOLD = 0.2
DEVICE = 'cuda:1'
PRESERVE = False  # False = ablate dog channels, True = preserve dog channels
PCT_CHANNELS = 9  #37 for Preserve, 9 for Ablate
MODE_TYPE = 'sum'  # 'top' or 'sum'
DATA_TYPE = 'contributions'  # 'contributions' or 'activations'
N_SUBSAMPLE= 50
USE_TOP5 = True


print("Loading ResNet50 model with FULL validation set...")
model, dataset, dataloader, layers_dict = bscope.ic.get_model(
    imagenet_path='/data/imagenet',
    which_model='resnet50',
    return_layers=True,
    batch_size=128,
    pin_memory=True,
    device=DEVICE,
    shuffle=False,
    subsample=N_SUBSAMPLE,
    subclasses=None
)
model.eval()

print(f"Loaded full dataset with {len(dataset)} images")

# ============================================================================
# GET DOG CONCEPT INDEX AND DOG CLASS INDICES
# ============================================================================
print("Loading dog concept...")
mask_matrix, mask_labels = bic.get_masks(
    path='/mnt/data/codec/hierarchy_metadata/pruned_hierarchy.json',
)

dog_concept_idx = mask_labels.index(CONCEPT_NAME)
dog_dataset_indices = np.where(mask_matrix[dog_concept_idx])[0]
dog_class_indices = np.unique(dog_dataset_indices // 50).astype(int)

print(f"Dog concept: {CONCEPT_NAME}")
print(f"Dog concept index in hierarchy: {dog_concept_idx}")
print(f"Dog ImageNet class indices: {dog_class_indices}")
print(f"Number of dog classes: {len(dog_class_indices)}")
print(f"\nGetting {MODE_TYPE} mode for concept '{CONCEPT_NAME}'...")

# Choose the right path based on data type
if DATA_TYPE == 'contributions':
    mode_path = ALEPH_CONTRIB_PATH
else:
    mode_path = ALEPH_ACTIV_PATH


# ============================================================================
# GET DOG MODE AND CHANNELS FROM MULTIPLE LAYERS
# ============================================================================
all_disruptors = []

for layer in LAYERS:
    print(f"\nProcessing Layer {layer}...")
    
    # Get the atom for this layer
    with h5.File(mode_path, 'r') as f:
        layer_key = str(layer)
        corr = f['layers'][layer_key]['corr_mtx'][:]
        dictionary = f['layers'][layer_key]['dictionary'][:]
        dog_corrs = corr[dog_concept_idx, :]
        
        if MODE_TYPE == 'top':
            top_mode_idx = np.argmax(dog_corrs)
            dog_atom = dictionary[top_mode_idx]
            n_modes = 1
        elif MODE_TYPE == 'sum':
            mode_idxs = np.where(dog_corrs > CORR_THRESHOLD)[0]
            if len(mode_idxs) == 0:
                top_mode_idx = np.argmax(dog_corrs)
                dog_atom = dictionary[top_mode_idx]
                n_modes = 1
            else:
                dog_atom = dictionary[mode_idxs].sum(axis=0)
                n_modes = len(mode_idxs)
    
    print(f"  {n_modes} mode(s), atom shape: {dog_atom.shape}")
    
    # Get channels to ablate
    num_chans = dog_atom.shape[0]
    num_to_keep = int(num_chans * PCT_CHANNELS / 100)
    top_channel_indices, _ = bic.top_n(dog_atom, num_to_keep)
    channels = list(top_channel_indices.astype(int))
    
    if PRESERVE:
        channels_to_ablate = list(set(range(num_chans)) - set(channels))
    else:
        channels_to_ablate = channels
    
    print(f"  Ablating {len(channels_to_ablate)} channels")
    
    # Create disruptor for this layer
    disruptor = bscope.Disruptor(layers_dict[layer], channels_to_ablate)
    all_disruptors.append(disruptor)

# ============================================================================
# CALCULATE ACCURACIES WITH MULTI-LAYER ABLATION
# ============================================================================
all_classes = list(range(1000))  
print("\nCalculating original accuracy...")
original_top1, original_top5 = bic.calculate_subsample_accuracy(
    model, dataloader, subclasses=all_classes, device=DEVICE
)

print("Ablating layers...")
for disruptor in all_disruptors:
    disruptor.activate()

disrupted_top1, disrupted_top5 = bic.calculate_subsample_accuracy(
    model, dataloader, subclasses=all_classes, device=DEVICE
)

# Deactivate all disruptors
for disruptor in all_disruptors:
    disruptor.deactivate()

print(f"Disrupted accuracy calculated: {len(disrupted_top1)} classes")



# ============================================================================
# SELECT WHICH ACCURACY TO PLOT
# ============================================================================
if USE_TOP5:
    original = original_top5
    disrupted = disrupted_top5
    acc_label = "Top-5"
else:
    original = original_top1
    disrupted = disrupted_top1
    acc_label = "Top-1"

# ============================================================================
# CALCULATE PERCENT CHANGE
# ============================================================================
epsilon = 1e-5
baseline = np.maximum(original, epsilon)
pct_change = (disrupted - original) / baseline * 100
pct_change = np.clip(pct_change, -100, 100)

# ============================================================================
# PLOT RESULTS
# ============================================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Plot 1: Original accuracy
axes[0].plot(original, 'ko', markersize=2)
axes[0].plot(dog_class_indices, original[dog_class_indices], 'bo', 
             markersize=4, label='types of dogs')
axes[0].set_ylabel(f'per-class\n{acc_label} accuracy')
axes[0].set_ylim([0, 120])
axes[0].legend(loc='upper left')
if PRESERVE:
    axes[0].set_title(f'preserve "{CONCEPT_NAME}" mode ({acc_label})')
else:
    axes[0].set_title(f'ablate "{CONCEPT_NAME}" mode ({acc_label})')

# Plot 2: Disrupted accuracy
axes[1].plot(disrupted, 'ro', markersize=2)
axes[1].plot(dog_class_indices, disrupted[dog_class_indices], 'bo', 
             markersize=4, label='types of dogs')
axes[1].set_ylabel(f'per-class\n{acc_label} accuracy')
axes[1].set_ylim([0, 120])

# Plot 3: Percent change
axes[2].plot(pct_change, 'k', linewidth=0.5)
axes[2].plot(dog_class_indices, pct_change[dog_class_indices], 'ro', 
             markersize=3)
axes[2].set_ylabel('% change')
axes[2].set_xlabel('imagenet classes')
axes[2].set_ylim([-100, 40])
axes[2].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()

F.XX(1.0, 1.0)
F.save(f'dog_ablate_iclr_{"preserve" if PRESERVE else "ablate"}')

# plt.show()
# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================
print(f"\n{'='*70}")
print("SUMMARY STATISTICS")
print(f"{'='*70}")
print(f"Total classes evaluated: {len(original_top1)}")
print(f"Average original accuracy: {np.mean(original_top1):.2f}%")
print(f"Average disrupted accuracy: {np.mean(disrupted_top1):.2f}%")
print(f"\nDog classes:")
print(f"  Original accuracy: {np.mean(original_top1[dog_class_indices]):.2f}%")
print(f"  Disrupted accuracy: {np.mean(disrupted_top1[dog_class_indices]):.2f}%")
print(f"  Average drop: {np.mean(original_top1[dog_class_indices] - disrupted_top1[dog_class_indices]):.2f}%")
print(f"\nNon-dog classes:")
non_dog_mask = np.ones(1000, dtype=bool)
non_dog_mask[dog_class_indices] = False
print(f"  Original accuracy: {np.mean(original_top1[non_dog_mask]):.2f}%")
print(f"  Disrupted accuracy: {np.mean(disrupted_top1[non_dog_mask]):.2f}%")
print(f"  Average drop: {np.mean(original_top1[non_dog_mask] - disrupted_top1[non_dog_mask]):.2f}%")