""" Train GP models on Niklas data with the CV criterion.

"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.update.universal_kriging import UniversalUpdatableGP
from volcapy.update.updatable_covariance import UpdatableGP
import volcapy.covariance.matern32 as kernel
from volcapy.grid.grid_from_dsm import Grid
from volcapy.universal_kriging.basis_functions import *
from torch.distributions.multivariate_normal import MultivariateNormal
from volcapy.utils import kMeans_folds

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
results_folder = "/storage/homefs/ct19x463/Dev/VolcapyProd/reporting/universal_kriging/results/Stromboli/"
os.makedirs(results_folder, exist_ok=True)


def main():
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

    # Build trends: constant.
    coeff_F = torch.ones(volcano_coords.shape[0], 1).float()

    # Model with trend.
    updatable_gp = UniversalUpdatableGP(kernel, lambda0_matern32, sigma0_matern32,
            volcano_coords,
            coeff_F, coeff_cov='uniform', coeff_mean='uniform',
            n_chunks=200)

    lambda0s = np.linspace(1.0, 600, 50)
    sigma0s = np.linspace(1.0, 600, 50)

    # Create folds.
    k = 10
    folds = kMeans_folds(k, data_coords)

    updatable_gp.train_cv_criterion(lambda0s, sigma0s, G, data_values, data_std,
            criterion = 'k fold', folds=folds,
            out_path=os.path.join(results_folder, "./train_res_cst_CV_10_folds_niklas.pkl"))


if __name__ == "__main__":
    main()
