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

output_path = "/home/ubuntu/Dev/VolcapyProd/reporting/sequential_ivr/results/myopic_173018"
data_folder = "/home/ubuntu/Dev/VolcapyProd/data/InversionDatas/stromboli_173018_cells"

# Indices of the data points that are along the coast.
from volcapy.data_preparation.paths import coast_data_inds


def main():
    os.makedirs(output_path, exist_ok=True)

    # Load
    F = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_full_surface.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()

    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"surface_data_coords.npy"))).float()

    ground_truth = torch.from_numpy(
            np.load(os.path.join(data_folder, "post_sample.npy")))
    data_values = torch.from_numpy(
            np.load(os.path.join(data_folder, "post_data_sample.npy")))

    # Dictionary between the original Niklas data and our discretization.
    niklas_data_inds = torch.from_numpy(
            np.load(os.path.join(data_folder, "niklas_data_inds_insurf.npy"))).long()
    niklas_coords = data_coords[niklas_data_inds].numpy()

    # PATHS ON THE VOLCANO.
    from volcapy.data_preparation.paths import paths as paths_niklas
    # Convert to indices in the full dataset.
    paths = []
    for path in paths_niklas:
        paths.append(niklas_data_inds[path].long())

    # --------------------------------
    # DEFINITION OF THE EXCURSION SET.
    # --------------------------------
    THRESHOLD_low = 700.0
    excursion_inds = (ground_truth >= THRESHOLD_low).nonzero()[:, 0]

    # Plot situation.
    plt.scatter(data_coords[:, 0], data_coords[:, 1], c="k", alpha=0.1)

    plt.scatter(
            volcano_coords[excursion_inds, 0],
            volcano_coords[excursion_inds, 1], c="r", alpha=0.07)

    plt.scatter(niklas_coords[:, 0], niklas_coords[:, 1],
            c=niklas_coords[:, 2])
    plt.scatter(niklas_coords[coast_data_inds, 0], niklas_coords[coast_data_inds, 1], c="r")

    for i, path in enumerate(paths):
        for x, y in zip(data_coords[path, 0], data_coords[path, 1]):
                plt.text(x, y, str(i), color="black", fontsize=6)

    plt.title("Paths on the Stromboli, location of coastal data and excursion set.")
    plt.show()

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
    from volcapy.strategy.myopic_weighted_ivr import MyopicStrategy
    strategy = MyopicStrategy(updatable_gp, data_coords,
            F, data_feed,
            lower=THRESHOLD_low, upper=None)

    start = timer()
    visited_inds, observed_data, ivrs = strategy.run(
            start_ind, n_steps=2000, data_std=0.1,
            output_folder=output_path, save_coverage=True)

    end = timer()
    print("Run in {} mins.".format((end - start)/60))

    np.save("visited_inds.npy", visited_inds)
    np.save("observed_data.npy", observed_data)
    np.save("ivrs.npy", ivrs)


if __name__ == "__main__":
    main()
