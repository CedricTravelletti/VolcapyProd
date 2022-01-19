""" Test the implementation of the computation of the inclusion probability.
First test on Jan. 19 2022.

"""
import os
import torch
import numpy as np
import volcapy.covariance.matern32 as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance import UpdatableGP
from volcapy.strategy.conservative_strategies import ConservativeStrategy


data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
ground_truth_folder = "/storage/homefs/ct19x463/AISTATS_results/final_samples_matern32/"
base_folder = "/storage/homefs/ct19x463/Conservative/test_inclusion/"


def main(sample_nr):
    # Create output directory.
    output_folder = os.path.join(base_folder,
            "wIVR_final_big/sample_{}".format(sample_nr))
    os.makedirs(output_folder, exist_ok=True)

    # Load static data.
    F = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_full_surface.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()
    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"surface_data_coords.npy"))).float()

    # Load generated data.
    ground_truth_path = os.path.join(ground_truth_folder,
            "prior_sample_{}.npy".format(sample_nr))
    ground_truth = torch.from_numpy(
            np.load(ground_truth_path))

    # Load prior realizations.
    N_REALIZATIONS = 100
    prior_realizations = []
    for i in range(100, 100 + N_REALIZATIONS):
        realization_path = os.path.join(ground_truth_folder,
            "prior_sample_{}.npy".format(i))
        prior_realizations.append(torch.from_numpy(
            np.load(realization_path)))

    # --------------------------------
    # DEFINITION OF THE EXCURSION SET.
    # --------------------------------
    THRESHOLD_low = 2500.0

    # Choose a starting points on the coast.
    start_ind = 4478
    
    # -------------------------------------
    # Define GP model.
    # -------------------------------------
    data_std = 0.1
    sigma0_matern32 = 284.66
    m0_matern32 = 2139.1
    lambda0_matern32 = 651.58

    # Prepare data.
    data_values = F @ ground_truth
    data_feed = lambda x: data_values[x]

    updatable_gp = UpdatableGP(cl, lambda0_matern32, sigma0_matern32,
            m0_matern32, volcano_coords, n_chunks=200)

    # -------------------------------------
    strategy = ConservativeStrategy(updatable_gp, data_coords,
            F, data_feed,
            lower=THRESHOLD_low, upper=None,
            prior_realizations=prior_realizations
            )

    # Run strategy.
    visited_inds, observed_data = strategy.run(
            start_ind, n_steps=4000, data_std=0.1,
            max_step=151.0,
            min_step=60.0,
            output_folder=output_folder
            )

if __name__ == "__main__":
    main(4)
