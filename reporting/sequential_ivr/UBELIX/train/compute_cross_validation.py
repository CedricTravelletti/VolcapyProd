""" Train hyperparameters.
"""
import os
import torch
import numpy as np

import volcapy.covariance.exponential as exponential_kernel
import volcapy.covariance.matern32 as matern32_kernel
import volcapy.covariance.matern52 as matern52_kernel

from volcapy.grid.grid_from_dsm import Grid
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess


data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
base_folder = "/storage/homefs/ct19x463/AISTATS_results/"


def main():
    # Load static data.
    F = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_corrected_final.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()

    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"niklas_data_coords_corrected_final.npy"))).float()
    data_values = torch.from_numpy(
            np.load(os.path.join(data_folder,"niklas_data_obs_corrected_final.npy"))).float()

    # Create a random shuffle of the data.
    inds_to_iter = list(range(data_values.shape[0]))
    shuffled_inds = np.random.shuffle(inds_to_iter)
    F_shuffled = F[shuffled_inds, :]
    data_values_shuffled = data_values[shuffled_inds]

    # Test/Train split.
    n_train = 300
    F_train = F_shuffled[:n_train, :]
    data_values_train = data_values_shuffled[:n_train]
    F_test = F_shuffled[n_train:, :]
    data_values_test = data_values_shuffled[n_train:]


    # HYPERPARAMETERS
    data_std = 0.1

    sigma0_exp = 308.89
    m0_exp = 535.39
    lambda0_exp = 1925.0

    sigma0_matern32 = 527.84
    m0_matern32 = 549.15
    lambda0_matern32 = 891.66

    sigma0_matern52 = 349.47
    m0_matern52 = 582.43
    lambda0_matern52 = 436.206

    gp_exp = InverseGaussianProcess(m0_exp, sigma0_exp, lambda0_exp,
            volcano_coords, exponential_kernel)
    gp_matern32 = InverseGaussianProcess(m0_matern32, sigma0_matern32,
            lambda0_matern32,
            volcano_coords, matern32_kernel)
    gp_matern52 = InverseGaussianProcess(m0_matern52, sigma0_matern52,
            lambda0_matern52,
            volcano_coords, matern52_kernel)

    # Condition on training data.
    m_post_m_exp, _  = gp_exp.condition_model(F_train, data_values_train, data_std)
    m_post_m_matern32, _  = gp_matern32.condition_model(F_train, data_values_train, data_std)
    m_post_m_matern52, _  = gp_matern52.condition_model(F_train, data_values_train, data_std)

    # Predict test data.
    data_values_pred_exp = F_test @ m_post_m_exp
    data_values_pred_matern32 = F_test @ m_post_m_matern32
    data_values_pred_matern52 = F_test @ m_post_m_matern52

    test_rmse_exp = torch.sqrt(torch.mean((data_values_test -
            data_values_pred_exp)**2))
    test_rmse_matern32 = torch.sqrt(torch.mean((data_values_test -
            data_values_pred_matern32)**2))
    test_rmse_matern52 = torch.sqrt(torch.mean((data_values_test -
            data_values_pred_matern52)**2))

    print("Test RMSE exponential: {}".format(test_rmse_exp.item()))
    print("Test RMSE Matern 3/2: {}".format(test_rmse_matern32.item()))
    print("Test RMSE Matern 5/2: {}".format(test_rmse_matern52.item()))


if __name__ == "__main__":
    main()
