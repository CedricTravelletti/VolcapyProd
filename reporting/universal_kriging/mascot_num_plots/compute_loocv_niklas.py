""" Compute leave-one-out cross-validation error on Niklas volcano data.

"""
import os
import torch
import numpy as np
import volcapy.covariance.matern32 as kernel
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance import UpdatableGP


data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
base_folder = "/storage/homefs/ct19x463/AISTATS_results/"
results_folder = "/storage/homefs/ct19x463/Dev/VolcapyProd/reporting/universal_kriging/mascot_num_plots/results/"


def main():
    # Load static data.
    G = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_corrected_final.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()

    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"niklas_data_coords_corrected_final.npy"))).float()
    data_values = torch.from_numpy(
            np.load(os.path.join(data_folder,"niklas_data_obs_corrected_final.npy"))).float()

    data_std = 0.1
    sigma0_matern32 = 284.66
    m0_matern32 = 2139.1
    lambda0_matern32 = 651.58

    constant_updatable_gp = UpdatableGP(kernel, lambda0_matern32, sigma0_matern32, m0_matern32,
            volcano_coords, n_chunks=200)
    residuals = constant_updatable_gp.leave_1_out_residuals(G, data_values, data_std)
    np.save(residuals.numpy(), os.path.join(results_folder, "./loocv_niklas.pck"))

if __name__ == "__main__":
    main()
