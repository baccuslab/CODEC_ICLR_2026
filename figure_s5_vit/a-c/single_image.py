# Apply standardized styling
import numpy as np
from scipy.stats import pearsonr
import bscope
import scipy.stats
import scipy.stats as stats
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from fycus import Fycus
from IPython import embed
import bscope.ic as bic
import os

cont_algo_str = "int_grad_top_1_True_steps_10"
model_type = 'vit'
layer_types = ['block', 'attention', 'mlp']
cont_spatial_operation = 'sum'  # 'sum', 'positive', 'negative'
act_spatial_operation = 'sum'  # 'sum', 'positive', 'negative'

layers = [9]
CLASS = 388
IDX = CLASS * 50 + 4  # Example index for a single image (class 388, image 4)

for layer_type in layer_types:
    for LAYER in layers:

        datapath = f'/data/codec/h5s/int_grad_top_1_True_{model_type}_{layer_type}_steps_10/data.h5'
        FIG = f'figure_s_vit_single_image_{cont_algo_str}_{model_type}_{layer_type}_cont_{cont_spatial_operation}_act_{act_spatial_operation}_l{LAYER}'
        F = Fycus('single_image', extension='svg')

        cont = bic.load_contribution_data(datapath, 'contributions', LAYER, cont_spatial_operation)[0][IDX]
        act = bic.load_contribution_data(datapath, 'activations', LAYER, act_spatial_operation)[0][IDX]

        plt.figure()
        plt.plot(act / np.std(act), 'b', lw=0.5, alpha=0.7, label='Activations')
        plt.plot(cont / np.std(cont), 'k', lw=0.5, alpha=0.7, label='Contributions')
        plt.xlabel('Channel')
        plt.ylabel('Normalized Value (std = 1)')
        plt.legend()
        plt.title(f'Single Image, Layer {LAYER}, {layer_type}')
        plt.xlim(0, len(act))
        F.QT()
        F.save(f'{FIG}')