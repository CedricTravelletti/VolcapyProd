""" Run myopic excursion set reconstruction strategy using weighted IVR.

"""
import os
import torch
import numpy as np
import volcapy.covariance.matern52 as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance import UpdatableGP
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud, _array_to_point_cloud
from volcapy.plotting.vtkutils import array_to_vector_cloud
import matplotlib.pyplot as plt

from timeit import default_timer as timer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_folder = "/home/ubuntu/Dev/VolcapyProd/data/InversionDatas/stromboli_173018_cells"
ground_truth_folder = "/home/ubuntu/Dev/VolcapyProd/data/AISTATS_results/"
output_folder = "/home/ubuntu/Dev/VolcapyProd/data/AISTATS_results/"


# Indices of the data points that are along the coast.
from volcapy.data_preparation.paths import coast_data_inds


def main(sample_nr):
    post_sample_path = os.path.join(ground_truth_folder,
            "post_samples/post_sample_{}.npy".format(sample_nr))
    post_data_sample_path = os.path.join(ground_truth_folder,
            "post_data_samples/post_data_sample_{}.npy".format(sample_nr))

    output_path = os.path.join(output_folder, "INFILL_results/sample_{}".format(sample_nr))
    os.makedirs(output_path, exist_ok=True)
    save_gp_state_path = os.path.join(output_path, "gp_state.pkl")

    # Load
    F = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_full_surface.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()

    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"surface_data_coords.npy"))).float()

    ground_truth = torch.from_numpy(
            np.load(post_sample_path))
    data_values = torch.from_numpy(
            np.load(post_data_sample_path))

    # Dictionary between the original Niklas data and our discretization.
    niklas_data_inds = torch.from_numpy(
            np.load(os.path.join(data_folder, "niklas_data_inds_insurf.npy"))).long()
    niklas_coords = data_coords[niklas_data_inds].numpy()

    # --------------------------------
    # DEFINITION OF THE EXCURSION SET.
    # --------------------------------
    THRESHOLD_low = 700.0
    excursion_inds = (ground_truth >= THRESHOLD_low).nonzero()[:, 0]

    # Coast data.
    coast_data_inds_infull = niklas_data_inds[coast_data_inds]

    # Choose a starting points on the coast.
    start_ind = coast_data_inds_infull[0]
    
    # Params
    data_std = 0.1
    # lambda0 = 338.0
    lambda0 = 338.46
    sigma0 = 359.49
    m0 = -114.40

    # Define GP model.
    data_feed = lambda x: data_values[x]
    updatable_gp = UpdatableGP(cl, lambda0, sigma0, m0, volcano_coords,
            n_chunks=80)
    from volcapy.strategy.infill import InfillStrategy
    strategy = InfillStrategy(updatable_gp, data_coords,
            F, data_feed,
            lower=THRESHOLD_low, upper=None,
            )

    start = timer()
    visited_inds, observed_data, ivrs = strategy.run(
            start_ind, n_steps=2000, data_std=0.1,
            output_folder=output_path, save_coverage=True,
            max_step = 310.0,
            save_gp_state_path=save_gp_state_path
            )

    end = timer()
    print("Run in {} mins.".format((end - start)/60))

    np.save(os.path.join(output_path, "visited_inds.npy"), visited_inds)
    np.save(os.path.join(output_path, "observed_data.npy"), observed_data)
    np.save(os.path.join(output_path, "ivrs.npy"), ivrs)


if __name__ == "__main__":
    main(4)
