""" Just test the inversion code on some known ground truth.

"""
import os
import torch
import numpy as np
import volcapy.covariance.matern52 as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance_nosigma import UpdatableGP
from volcapy.strategy.myopic_ivr import MyopicIVRStrategy

from timeit import default_timer as timer


data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_40357_cells/"
ground_truth_path = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_40357_cells/prior_sample.npy"


def main():
    # Load static data.
    F = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_full_surface.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()
    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"surface_data_coords.npy"))).float()

    # Load generated data.
    ground_truth = torch.from_numpy(
            np.load(ground_truth_path))

    # --------------------------------
    # DEFINITION OF THE EXCURSION SET.
    # --------------------------------
    THRESHOLD_low = 700.0

    # Choose a starting points on the coast.
    start_ind = 4478
    
    # -------------------------------------
    # Define GP model.
    # -------------------------------------
    data_std = 0.1
    lambda0 = 338.46
    sigma0 = 359.49
    m0 = -114.40

    # Prepare data.
    data_values = F @ ground_truth

    data_feed = lambda x: data_values[x]
    updatable_gp = UpdatableGP(cl, lambda0, sigma0, m0, volcano_coords,
            n_chunks=200)

    updatable_gp.update(F[1].reshape(1, -1), data_values[1], 0.1)
