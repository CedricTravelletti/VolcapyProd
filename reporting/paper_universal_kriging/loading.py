""" Convenience function for loading Niklas volcano data.

"""

import os
import numpy as np
import torch
from volcapy.grid.grid_from_dsm import Grid


def load_niklas_volcano_data(base_folder):
    data_folder = os.path.join(base_folder, "InversionDatas/stromboli_173018")

    G = (
        torch.from_numpy(np.load(os.path.join(data_folder, "F_corrected_final.npy")))
        .float()
        .detach()
    )
    grid = Grid.load(os.path.join(data_folder, "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()

    y_coords = torch.from_numpy(
        np.load(os.path.join(data_folder, "niklas_data_coords_corrected_final.npy"))
    ).float()
    y = torch.from_numpy(
        np.load(os.path.join(data_folder, "niklas_data_obs_corrected_final.npy"))
    ).float()
    y_std = 0.1
    return {
        "G": G,
        "grid": grid,
        "volcano_coords": volcano_coords,
        "y_coords": y_coords,
        "y": y,
        "y_std": y_std,
    }
