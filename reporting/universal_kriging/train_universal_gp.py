""" Train a universal GP on the log-gaussian comparison data.

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

    # Define GP model.
    data_std = 0.1
    sigma0 = 1.0
    m0 = 2139.1
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

    ground_truth = torch.from_numpy(np.load(os.path.join(results_folder, "ground_truth.npy")))
    synth_data = torch.from_numpy(
            np.load(os.path.join(results_folder, "synth_data.npy")))
    lambda0s = np.linspace(1.0, 3000, 30)
    kappa_s = np.linspace(1e-5, 1, 30)

    updatable_gp.train(lambda0s, kappa_s, G, synth_data,
            out_path=os.path.join(results_folder, "./train_res_universal.pck"))


if __name__ == "__main__":
    main()
