import bscope.ic as bic
import bscope
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from fycus import Fycus
import pickle
import os

F = Fycus('hyperparam', base_path='/home/zalaoui/higanbana/STSAE')

# ============================================================================
# CONFIGURATION
# ============================================================================
ALEPH_CONTRIB_PATH = '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/aleph_contributions_positive/mode_summary.h5'
ALEPH_ACTIV_PATH = '/mnt/data/codec/h5s/int_grad_top_1_False_resnet50_steps_10/saes/aleph_activations_positive/mode_summary.h5'

CONCEPT_NAME = 'dog.n.01'  # The concept
LAYERS = [12, 13, 14, 15]  # List of layers to ablate
CORR_THRESHOLD = 0.2
DEVICE = 'cuda:1'
MODE_TYPE = 'sum'  # 'top' or 'sum'
DATA_TYPE = 'contributions'  # 'contributions' or 'activations'
N_SUBSAMPLE = 50
USE_TOP5 = True

PKL_DIR = os.path.dirname(os.path.abspath(__file__))

# Set COMPUTE = True to run inference and save pkl files.
# Set COMPUTE = False to skip inference and go straight to plotting.
COMPUTE = False

# Each entry: (PRESERVE, PCT_CHANNELS, label)
CONFIGS = [
    (False, 9,  'ablate'),
    (True,  37, 'preserve'),
]


# ============================================================================
# HELPERS
# ============================================================================
def get_channels_to_ablate(mode_path, layer, dog_concept_idx, pct_channels, preserve):
    with h5.File(mode_path, 'r') as f:
        layer_key = str(layer)
        corr = f['layers'][layer_key]['corr_mtx'][:]
        dictionary = f['layers'][layer_key]['dictionary'][:]
        dog_corrs = corr[dog_concept_idx, :]

        mode_idxs = np.where(dog_corrs > CORR_THRESHOLD)[0]
        if len(mode_idxs) == 0:
            top_mode_idx = np.argmax(dog_corrs)
            dog_mode = dictionary[top_mode_idx]
            n_modes = 1
        else:
            dog_mode = dictionary[mode_idxs].sum(axis=0)
            n_modes = len(mode_idxs)

    num_chans = dog_mode.shape[0]
    num_to_keep = int(num_chans * pct_channels / 100)
    top_channel_indices, _ = bic.top_n(dog_mode, num_to_keep)
    channels = list(top_channel_indices.astype(int))

    if preserve:
        channels_to_ablate = list(set(range(num_chans)) - set(channels))
    else:
        channels_to_ablate = channels

    return channels_to_ablate, n_modes


def pkl_path(label):
    return os.path.join(PKL_DIR, f'dog_{label}_results.pkl')


# ============================================================================
# COMPUTE SECTION — runs both configs and saves pkl files
# ============================================================================
if COMPUTE:
    # Choose mode path
    mode_path = ALEPH_CONTRIB_PATH if DATA_TYPE == 'contributions' else ALEPH_ACTIV_PATH

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

    # Get dog concept info (shared across configs)
    print("Loading dog concept...")
    mask_matrix, mask_labels = bic.get_masks(
        path='/mnt/data/codec/hierarchy_metadata/pruned_hierarchy.json',
    )
    dog_concept_idx = mask_labels.index(CONCEPT_NAME)
    dog_dataset_indices = np.where(mask_matrix[dog_concept_idx])[0]
    dog_class_indices = np.unique(dog_dataset_indices // 50).astype(int)

    print(f"Dog concept: {CONCEPT_NAME}")
    print(f"Dog ImageNet class indices: {dog_class_indices}")
    print(f"Number of dog classes: {len(dog_class_indices)}")

    # Compute baseline accuracy once
    all_classes = list(range(1000))
    print("\nCalculating original accuracy...")
    original_top1, original_top5 = bic.calculate_subsample_accuracy(
        model, dataloader, subclasses=all_classes, device=DEVICE
    )

    # Run each config
    for (preserve, pct_channels, label) in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Running config: {label} (PRESERVE={preserve}, PCT_CHANNELS={pct_channels})")
        print(f"{'='*60}")

        all_disruptors = []
        for layer in LAYERS:
            print(f"  Processing Layer {layer}...")
            channels_to_ablate, n_modes = get_channels_to_ablate(
                mode_path, layer, dog_concept_idx, pct_channels, preserve
            )
            print(f"    {n_modes} mode(s), ablating {len(channels_to_ablate)} channels")
            disruptor = bscope.Disruptor(layers_dict[layer], channels_to_ablate)
            all_disruptors.append(disruptor)

        print("  Activating disruptors...")
        for d in all_disruptors:
            d.activate()

        disrupted_top1, disrupted_top5 = bic.calculate_subsample_accuracy(
            model, dataloader, subclasses=all_classes, device=DEVICE
        )

        for d in all_disruptors:
            d.deactivate()

        # Save to pkl
        results = {
            'original_top1': original_top1,
            'original_top5': original_top5,
            'disrupted_top1': disrupted_top1,
            'disrupted_top5': disrupted_top5,
            'dog_class_indices': dog_class_indices,
            'preserve': preserve,
            'pct_channels': pct_channels,
            'label': label,
        }
        out_path = pkl_path(label)
        with open(out_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"  Saved results to {out_path}")


# ============================================================================
# PLOT SECTION — loads pkl files and plots side by side
# ============================================================================
print("\nLoading pkl files for plotting...")
all_results = {}
for (_, _, label) in CONFIGS:
    with open(pkl_path(label), 'rb') as f:
        all_results[label] = pickle.load(f)
    print(f"  Loaded {pkl_path(label)}")

fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

acc_label = "Top-5" if USE_TOP5 else "Top-1"

TICK_SIZE  = 18
LABEL_SIZE = 16
TITLE_SIZE = 18

for col, (_, _, label) in enumerate(CONFIGS):
    r = all_results[label]
    dog_class_indices = r['dog_class_indices']

    original  = r['original_top5']  if USE_TOP5 else r['original_top1']
    disrupted = r['disrupted_top5'] if USE_TOP5 else r['disrupted_top1']

    epsilon = 1e-5
    pct_change = (disrupted - original) / np.maximum(original, epsilon) * 100
    pct_change = np.clip(pct_change, -100, 100)

    mode_str = 'preserve dog mode' if r['preserve'] else 'ablate dog mode'

    # Row 0: original accuracy
    axes[0, col].plot(original, 'ko', markersize=2)
    axes[0, col].plot(dog_class_indices, original[dog_class_indices], 'bo',
                      markersize=4, label='types of dogs')
    axes[0, col].set_ylabel(f'per-class\n{acc_label} accuracy', fontsize=LABEL_SIZE)
    axes[0, col].set_ylim([0, 120])
    axes[0, col].set_title(mode_str, fontsize=TITLE_SIZE)
    leg = axes[0, col].legend(loc='lower left', bbox_to_anchor=(0.0, 1.01),
                              borderaxespad=0, frameon=False, fontsize=LABEL_SIZE)
    for text in leg.get_texts():
        text.set_color('blue')

    # Row 1: disrupted accuracy
    axes[1, col].plot(disrupted, 'ro', markersize=2)
    axes[1, col].plot(dog_class_indices, disrupted[dog_class_indices], 'bo',
                      markersize=4, label='types of dogs')
    axes[1, col].set_ylabel(f'per-class\n{acc_label} accuracy', fontsize=LABEL_SIZE)
    axes[1, col].set_ylim([0, 120])

    # Row 2: percent change
    axes[2, col].plot(pct_change, 'k', linewidth=0.5)
    axes[2, col].plot(dog_class_indices, pct_change[dog_class_indices], 'ro', markersize=3)
    axes[2, col].set_ylabel('% change', fontsize=LABEL_SIZE)
    axes[2, col].set_xlabel('imagenet classes', fontsize=LABEL_SIZE)
    axes[2, col].set_ylim([-100, 40])
    axes[2, col].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    for row in range(3):
        axes[row, col].tick_params(axis='both', labelsize=TICK_SIZE)

plt.tight_layout()

F.XX(1.0, 1.5)
F.save('dog_ablate_iclr_combined')

