""" Run the random walk strategy.

"""
import os
import torch
import numpy as np
import volcapy.covariance.matern52 as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance import UpdatableGP
from volcapy.strategy.random_walk import RandomWalkStrategy

from timeit import default_timer as timer


data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
ground_truth_folder = "/storage/homefs/ct19x463/AISTATS_results/"
base_folder = "/storage/homefs/ct19x463/AISTATS_results/"


def main(sample_nr):
    # Create output directory.
    output_folder = os.path.join(base_folder, "RANDOMWALK_results/sample_{}".format(sample_nr))
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
            "post_samples/post_sample_{}.npy".format(sample_nr))
    post_data_sample_path = os.path.join(ground_truth_folder,
            "post_data_samples/post_data_sample_{}.npy".format(sample_nr))
    ground_truth = torch.from_numpy(
            np.load(post_sample_path))
    data_values = torch.from_numpy(
            np.load(post_data_sample_path))

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

    data_feed = lambda x: data_values[x]
    updatable_gp = UpdatableGP(cl, lambda0, sigma0, m0, volcano_coords,
            n_chunks=80)

    # -------------------------------------

    from volcapy.strategy.random_walk import RandomWalkStrategy
    strategy = RandomWalkStrategy(updatable_gp, data_coords,
            F, data_feed,
            lower=THRESHOLD_low, upper=None,
            )

    start = timer()
    # Run strategy.
    visited_inds, observed_data = strategy.run(
            start_ind, n_steps=2000, data_std=0.1,
            output_folder=output_folder, save_coverage=True,
            max_step = 310.0,
            save_gp_state_path=save_gp_state_path
            )

    end = timer()
    print("Run in {} mins.".format((end - start)/60))

if __name__ == "__main__":
    main(5)
