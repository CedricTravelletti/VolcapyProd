""" Analyze the results of the myopic run. I.e. mem.usage, reconstruction
quality, and so on.

"""
import os
import torch
import numpy as np
import volcapy.covariance.matern52 as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance import UpdatableGP

from timeit import default_timer as timer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

output_path = "/home/ubuntu/Dev/VolcapyProd/reporting/sequential_ivr/results/analysis"
data_folder = "/home/ubuntu/Dev/VolcapyProd/data/InversionDatas/stromboli_173018_cells"
results_folder = "/home/ubuntu/Dev/VolcapyProd/reporting/sequential_ivr/results/myopic_173018"

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

    # --------------------------------
    # DEFINITION OF THE EXCURSION SET.
    # --------------------------------
    THRESHOLD_low = 700.0
    excursion_inds = (ground_truth >= THRESHOLD_low).nonzero()[:, 0]
    
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

    # Asimilate myopic data collection plan.
    visited_inds = np.load(os.path.join(results_folder, "visited_inds.npy"))
    observed_data = np.load(os.path.join(results_folder, "visited_inds.npy"))

    n_chunks = 80
    for i, inds in enumerate(np.array_split(visited_inds, n_chunks)):
        print("Processing chunk {} / {}".format(i, n_chunks))
        y = data_feed(inds)
        G_current = F[inds, :]
        updatable_gp.update(G_current, y, data_std)

if __name__ == "__main__":
    main()
