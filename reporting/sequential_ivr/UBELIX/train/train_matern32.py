""" Train hyperparameters.
"""
import os
import torch
import numpy as np
import volcapy.covariance.matern32 as kernel
from volcapy.grid.grid_from_dsm import Grid
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess


data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
original_data_folder = "/storage/homefs/ct19x463/Data/original/"
base_folder = "/storage/homefs/ct19x463/AISTATS_results/"


def main():
    # Load static data.
    F = torch.from_numpy(
            np.load(os.path.join(original_data_folder, "F_niklas.npy"))).float().detach()

    """
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    """
    volcano_coords = torch.from_numpy(
            np.load(os.path.join(original_data_folder, "volcano_coords"))).float().detach()

    data_coords = torch.from_numpy(
            np.load(os.path.join(original_data_folder,"data_coords.npy"))).float().detach()
    data_values = torch.from_numpy(
            np.load(os.path.join(original_data_folder,"data_obs.npy"))).float().detach()


    print("Size of inversion grid: {} cells.".format(volcano_coords.shape[0]))
    print("Number of datapoints: {}.".format(data_coords.shape[0]))
    size = data_coords.shape[0]*volcano_coords.shape[0]*4 / 1e9
    cov_size = (volcano_coords.shape[0])**2 * 4 / 1e9
    print("Size of Covariance matrix: {} GB.".format(cov_size))
    print("Size of Pushforward matrix: {} GB.".format(size))

    # HYPERPARAMETERS to start search from.
    data_std = 0.1
    sigma0 = 300.0
    m0 = 561.0
    lambda0 = 50.0

    myGP = InverseGaussianProcess(m0, sigma0, lambda0,
            volcano_coords, kernel)

    # Train.
    myGP.train(np.linspace(20.0, 1020.0, 20), F, data_values, data_std,
            out_path="./train_res_matern32.pck",
            n_epochs=3000, lr=0.5,
            n_chunks=80,
            n_flush=1)


if __name__ == "__main__":
    main()
