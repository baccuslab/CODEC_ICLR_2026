# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

Figure-generation code for the CODEC ICLR 2026 paper. Each top-level directory corresponds to one figure and follows the convention from README.md:
- `scripts/` — Python scripts that generate the figure
- `panels/` — saved `.svg` panels composing the figure
- a final `.svg` assembling the panels

## Running Scripts

Scripts are run directly with Python (no build system or test suite):

```bash
python visualization/scripts/visualization_2.py
python mode_stats/scripts/class_coverage.py
```

Scripts require GPU (`cuda:0` or `cuda:1`) and access to data at `/data/`.

## Key Internal Libraries

- **`bscope` / `bscope.ic` (as `bic`)** — model inspection toolkit. Core API:
  - `bic.ModeSummary(path)` — loads pre-computed SAE mode summaries from an H5 file
  - `bic.get_top_mode(mode_summary, layer, class_idx, which_mode)` → `(mode_idx, atom, mode_loadings, corr)`
  - `bic.top_n(atom, n)` → `(chan_idxs, vals)` — top-N channels by magnitude
  - `bscope.ic.get_model('resnet50', return_layers=True, imagenet_path=...)` → `(model, dataset, dataloader, layers)`
  - `bscope.Inspector([layer], to_numpy=False)` — hooks into model to capture `.activations`
  - `bic.normalize(x)`, `bic.normalize_symmetric(x)` — image normalization helpers

- **`fycus`** — figure output manager. Usage pattern:
  ```python
  F = fycus.Fycus('output_name')  # sets output directory / name
  F.XX(width, height)              # set figure size
  F.save('filename', dpi=150)     # saves SVG panel
  ```

- **`bopt`** — imported in all scripts; utility library (specific usage not yet characterized).

## Data Paths

- ImageNet: `/data/imagenet/`
- Mode summary H5 files: `/data/h5s/<attribution_method>/saes/aleph_contributions_positive/mode_summary.h5`
  - `int_grad_top_1_False_resnet50_steps_10` — integrated gradients attribution
  - `act_normgrad_top_1_False_resnet50` — activation × normalized-gradient attribution

## Core Algorithm Pattern (visualization scripts)

1. Load `ModeSummary` → get top mode atom for `(LAYER, CLASS, WHICH_MODE)` → select top-N channels
2. Forward pass through ResNet50, capture activations at `layers[LAYER]` via `Inspector`
3. Compute `Jy` = ∂(class score) / ∂(activations), `Jz` = ∂(activation[k,h,w]) / ∂(input image X)
4. Combined Jacobian: `J = einsum('abcd,a->abcd', Jz, Jy)` per spatial/channel position
5. Contribution scalar: `C = einsum('abcd,abcd->a', J, X)` — split into positive/negative
6. Accumulate into `PossumJyJz` / `NegsumJyJz`, then elementwise-multiply with X to get masked images
7. Apply `median_filter`, clip, and visualize with matplotlib; save via `F.save()`

Key configuration variables at top of each script: `LAYER`, `CLASS`, `NCHAN` (number of top channels), `WHICH_MODE`, `filter_size`, `contrast`, `use_norm`.
