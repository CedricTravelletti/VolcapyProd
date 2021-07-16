""" Run the random walk strategy.

"""
import os
import torch
import numpy as np
import volcapy.covariance.matern32 as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance import UpdatableGP
from volcapy.strategy.myopic_weighted_ivr import MyopicWIVRStrategy

from timeit import default_timer as timer


data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
ground_truth_folder = "/storage/homefs/ct19x463/AISTATS_results/final_samples_matern32/"
base_folder = "/storage/homefs/ct19x463/AISTATS_results/"


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
    post_sample_path = os.path.join(ground_truth_folder,
            "prior_sample_{}.npy".format(sample_nr))
    ground_truth = torch.from_numpy(
            np.load(post_sample_path))

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
    updatable_gp = UpdatableGP(cl, lambda0_matern32, sigma0_matern32, m0_matern32, volcano_coords,
            n_chunks=200)

    # -------------------------------------

    from volcapy.strategy.random_walk import RandomWalkStrategy
    strategy = MyopicWIVRStrategy(updatable_gp, data_coords,
            F, data_feed,
            lower=THRESHOLD_low, upper=None,
            )

    start = timer()
    # Run strategy.
    visited_inds, observed_data = strategy.run(
            start_ind, n_steps=4000, data_std=0.1,
            max_step=151.0,
            min_step=60.0,
            output_folder=output_folder,
            restart_from_save=output_folder
            )

    end = timer()
    print("Run in {} mins.".format((end - start)/60))

if __name__ == "__main__":
    main(5)
