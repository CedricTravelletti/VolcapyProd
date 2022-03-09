""" Test universal kriging on the Stromboli volcano. And compare it to a constant model.

This script runs tests in a well specified setting, i.e. 
we generate random volcanoes by drawing from the prior 
and the run the inversion on those ground truths.

"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.update.universal_kriging import UniversalUpdatableGP
from volcapy.update.updatable_covariance import UpdatableGP
import volcapy.covariance.matern52 as kernel
from volcapy.grid.grid_from_dsm import Grid
from volcapy.universal_kriging.basis_functions import *
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np
import os
import sys

# Now torch in da place.
import torch

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')

data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018/"
# data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
# results_folder = "/storage/homefs/ct19x463/UniversalKrigingResults/WellSpecified"

# ground_truth_folder = "/storage/homefs/ct19x463/AISTATS_results/final_samples_matern32/"
# base_folder = "/storage/homefs/ct19x463/AISTATS_results/"


def main():
    # Load
    G = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_niklas.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(
            grid.cells).float().detach()
    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"niklas_data_coords.npy"))).float()
    data_values = torch.from_numpy(
            np.load(os.path.join(data_folder,"niklas_data_obs.npy"))).float()
    n_data = G.shape[0]

    # Define GP model.
    data_std = 0.1
    sigma0 = 284.66
    m0 = 2139.1
    lambda0 = 651.58

    # Build trends: constant + planar + cylindrical.
    x0 = volcano_coords[:, 0].mean() # Volcano center.
    y0 = volcano_coords[:, 1].mean()
    z0 = volcano_coords[:, 2].mean()

    coeff_mean = torch.tensor([m0, 0.0, 0.0]).reshape(-1, 1)
    coeff_cov = torch.tensor([
            [200.0, 0, 0],
            [0, 0.05, 0],
            [0, 0, 0.05]
            ])
    coeff_F = torch.hstack([
        torch.ones(volcano_coords.shape[0], 1),
        planar(
            volcano_coords,
            x0, y0, z0, phi=torch.tensor([45]), theta=torch.tensor([45])).reshape(-1, 1),
        cylindrical(
            volcano_coords,
            x0, y0).reshape(-1, 1)
        ])

    # Model with trend.
    updatable_gp = UniversalUpdatableGP(kernel, lambda0, torch.tensor([sigma0]),
            volcano_coords,
            coeff_F, coeff_cov, coeff_mean,
            n_chunks=200)

    # Sample artificial volcano and 
    # invert data at Niklas points.
    ground_truth, true_trend_coeffs = updatable_gp.sample_prior()
    noise = MultivariateNormal(loc=torch.zeros(n_data), covariance_matrix=data_std**2 * torch.eye(n_data)).rsample()
    synth_data = G @ ground_truth + noise
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


if __name__ == "__main__":
    main()
