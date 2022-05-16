""" Test universal kriging on the Stromboli volcano. And compare it to a constant model.

This script runs the full inversion loop, that is hyperparameter training and data assimilation 
on a synthetic ground truth consisting of a log-gaussian random field and a cylindrical trend.

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
results_folder ="/home/cedric/PHD/Dev/VolcapySIAM/reporting/universal_kriging/results/const_vs_univ_loggaussian/"
os.makedirs(results_folder, exist_ok=True)
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
    # sigma0 = 284.66
    sigma0 = 1.0
    m0 = 2139.1
    # lambda0 = 651.58
    lambda0 = 200.0

    # Build trends: constant cylindrical.
    x0 = volcano_coords[:, 0].mean() # Volcano center.
    y0 = volcano_coords[:, 1].mean()
    z0 = volcano_coords[:, 2].mean()

    coeff_mean = torch.tensor([m0, 0.01]).reshape(-1, 1).float()
    coeff_cov = torch.tensor([
            [200.0, 0],
            [0, 0.05]
            ]).float()
    coeff_F = torch.hstack([
        torch.ones(volcano_coords.shape[0], 1),
        cylindrical(
            volcano_coords,
            x0, y0).reshape(-1, 1)
        ]).float()

    # Model with trend.
    updatable_gp = UniversalUpdatableGP(kernel, lambda0, torch.tensor([sigma0]),
            volcano_coords,
            coeff_F, coeff_cov, coeff_mean,
            n_chunks=200)

    # Sample artificial log-normal volcano.
    gp_sampler = UpdatableGP(kernel, lambda0, torch.tensor([sigma0]), 0,
            volcano_coords, n_chunks=200)

    """
    # Add trend to generate ground truth.
    # Commented out since we re-use an already nice looking one.
    ground_truth_no_trend = torch.exp(gp_sampler.sample_prior())
    true_trend = coeff_F @ coeff_mean
    ground_truth = ground_truth_no_trend + true_trend
    np.save(os.path.join(results_folder, "ground_truth.npy"), ground_truth.cpu().numpy())
    """
    ground_truth = torch.from_numpy(np.load(os.path.join(results_folder, "ground_truth.npy")))

    # Add noise and generate data.
    """
    noise = MultivariateNormal(loc=torch.zeros(n_data), covariance_matrix=data_std**2 * torch.eye(n_data)).rsample()
    synth_data = G @ ground_truth + noise
    np.save(os.path.join(results_folder, "synth_data.npy"), synth_data.cpu().numpy())
    """
    synth_data = np.load(os.path.join(results_folder, "synth_data.npy"))

    # Now train GP model on it.
    constant_updatable_gp = UpdatableGP(kernel, lambda0, torch.tensor([sigma0]), m0_true,
            volcano_coords, n_chunks=200)


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
