""" Run the random walk strategy.

"""
import os
import torch
import numpy as np
import volcapy.covariance.matern52 as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance import UpdatableGP
from volcapy.strategy.infill import InfillStrategy

from timeit import default_timer as timer


data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
ground_truth_folder = "/storage/homefs/ct19x463/AISTATS_results/prior_samples_April2021"
base_folder = "/storage/homefs/ct19x463/AISTATS_results/"


def main(sample_nr):
    # Create output directory.
    output_folder = os.path.join(base_folder,
            "INFILL_results_350_nonoise/prior_samples_April2021/sample_{}".format(sample_nr))
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
    THRESHOLD_low = 350.0
    true_excursion_inds = (ground_truth >= THRESHOLD_low).nonzero()[:, 0]
    print("True excursion size: {} cells.".format(true_excursion_inds.shape[0]))

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

    # -------------------------------------
    strategy = InfillStrategy(updatable_gp, data_coords,
            F, data_feed,
            lower=THRESHOLD_low, upper=None,
            )

    start = timer()
    # Run strategy.
    visited_inds, observed_data = strategy.run(
            data_std=0.1,
            output_folder=output_folder,
            n_data_splits=200
            )

    end = timer()
    print("Run in {} mins.".format((end - start)/60))

if __name__ == "__main__":
    main(8)
