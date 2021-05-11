""" Train hyperparameters.
"""
import os
import torch
import numpy as np
import volcapy.covariance.matern52 as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance import UpdatableGP
from volcapy.strategy.myopic_weighted_ivr import MyopicWIVRStrategy

from timeit import default_timer as timer


data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
base_folder = "/storage/homefs/ct19x463/AISTATS_results/"


def main():
    # Create output directory.
    output_folder = os.path.join(base_folder, "./train_res.pck")

    # Load static data.
    F = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_full_surface.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()
    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"surface_data_coords.npy"))).float()

    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"niklas_data_coords.npy"))).float()
    data_values = torch.from_numpy(
            np.load(os.path.join(data_folder,"niklas_data_obs.npy"))).float()


    print("Size of inversion grid: {} cells.".format(volcano_coords.shape[0]))
    print("Number of datapoints: {}.".format(data_coords.shape[0]))
    size = data_coords.shape[0]*volcano_coords.shape[0]*4 / 1e9
    cov_size = (volcano_coords.shape[0])**2 * 4 / 1e9
    print("Size of Covariance matrix: {} GB.".format(cov_size))
    print("Size of Pushforward matrix: {} GB.".format(size))

    # HYPERPARAMETERS to start search from.
    data_std = 0.1
    sigma0 = 100.6
    m0 = 1000.0
    lambda0 = 300.0

    myGP = InverseGaussianProcess(m0, sigma0, lambda0,
            volcano_coords, kernel,
            logger=logger)

    # Train.
    myGP.train(np.linspace(20, 1400, 14), F, data_values, data_std,
            out_path="./train_res.pck",
            n_chunks=100,
            n_flush=1)


if __name__ == "__main__":
    main()
