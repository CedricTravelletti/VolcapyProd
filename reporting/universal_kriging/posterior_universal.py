""" Study the posterior for the universal GP (in the log-gaussian comparison).
"""
from volcapy.update.universal_kriging import UniversalUpdatableGP
import volcapy.covariance.matern52 as kernel
from volcapy.grid.grid_from_dsm import Grid
from volcapy.universal_kriging.basis_functions import *

import numpy as np
import os
import sys

# Now torch in da place.
import torch

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')

# data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018/"
# results_folder ="/home/cedric/PHD/Dev/VolcapySIAM/reporting/universal_kriging/results/const_vs_univ_loggaussian/"
data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
results_folder = "/storage/homefs/ct19x463/Dev/VolcapyProd/reporting/universal_kriging/results/const_vs_univ_loggaussian/"
os.makedirs(results_folder, exist_ok=True)

def main():
    # Load
    G = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_niklas.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(
            grid.cells).float().detach()

    # Optimal hyperparameters.
    data_std = 0.1
    sigma0 = 2.968352
    lambda0 = 207.8275

    ground_truth = torch.from_numpy(np.load(os.path.join(results_folder, "ground_truth.npy")))
    synth_data = torch.from_numpy(
            np.load(os.path.join(results_folder, "synth_data.npy")))

    # Build trends: constant cylindrical.
    x0 = volcano_coords[:, 0].mean() # Volcano center.
    y0 = volcano_coords[:, 1].mean()
    z0 = volcano_coords[:, 2].mean()
    coeff_F = torch.hstack([
        torch.ones(volcano_coords.shape[0], 1),
        cylindrical(
            volcano_coords,
            x0, y0).reshape(-1, 1)
        ]).float()

    # Now fit GP model
    updatable_gp = UniversalUpdatableGP(kernel, lambda0, torch.tensor([sigma0]),
            volcano_coords,
            coeff_F, coeff_cov="uniform", coeff_mean="uniform",
            n_chunks=200)

    # Compute cross-validation matrix.
    K_tilde = updatable_gp.compute_cv_matrix(G, synth_data, data_std=0.2)
    np.save(os.path.join(results_folder, "K_tilde.npy"),
            K_tilde.cpu().numpy())

    # Compute posterior mean.
    updatable_gp.update_uniform(G, synth_data, data_std)
    np.save(os.path.join(results_folder, "post_mean_universal.npy"),
            updatable_gp.post_mean.cpu().numpy())

    np.save(os.path.join(results_folder, "coeff_post_mean.npy"),
            updatable_gp.coeff_post_mean.cpu().numpy())

    variance = updatable_gp.covariance.extract_variance()
    np.save(os.path.join(results_folder, "variance_universal.npy"),
            variance.cpu().numpy())

    """
    updatable_gp.update(G, synth_data, data_std)
    np.save("post_mean_universal.npy", updatable_gp.mean_vec.detach().cpu().numpy())
    np.save("ground_truth.npy", ground_truth.cpu().numpy())

    # Model who thinks the trend is a constant.
    # Let's be fair and allow it to know the true mean.
    m0_true = true_trend_coeffs[0]
    constant_updatable_gp = UpdatableGP(kernel, lambda0, torch.tensor([sigma0]), m0_true,
            volcano_coords, n_chunks=200)
    constant_updatable_gp.update(G, synth_data, data_std)
    np.save("post_mean_constant.npy", constant_updatable_gp.mean_vec.detach().cpu().numpy())
    np.save("true_trend_coeffs.npy", true_trend_coeffs.detach().cpu().numpy())
    np.save("trend_matrix.npy", coeff_F.detach().cpu().numpy())
    """


if __name__ == "__main__":
    main()
