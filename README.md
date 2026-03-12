# CODEC_ICLR_2026

To regenerate results from Melander, Alaoui, Liu, Ganguli, and Baccus (2026):
1. Compute and save-out per-input contributions into an h5 file
2. Train SAE on h5 file and generate a mode-summary that correlates modes (atoms) with semantic categories
3. Use resulting data and models to generate figures in each sub-directory


### Dependencies
- Beyond the standard scientific stack, we rely heavily on 
  - [`bscope`](https://github.com/baccuslab/bscope): for acquiring, saving, and manipulating the intermediate activations and contributions of a given model  
  - [`fycus`](https://github.com/jbmelander/fycus): a little plotting library to help with visualization. This can simply be commented out. Replace all Fycus.save('...') calls with plt.imshow() and you can generate the same figures via standard means.

