""" Train GP with cross-validation with k means folds.
"""
from volcapy.update.universal_kriging import UniversalUpdatableGP
import volcapy.covariance.matern52 as kernel
from volcapy.grid.grid_from_dsm import Grid
from volcapy.universal_kriging.basis_functions import *
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
    data_coords = np.load(os.path.join(data_folder,
                    "niklas_data_coords.npy"))

    data_std = 0.1
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

    # Define GP model with arbitrary parameters (we will train them anyway).
    lambda0, sigma0 = 10, 2
    updatable_gp = UniversalUpdatableGP(kernel, lambda0, sigma0,
            volcano_coords,
            coeff_F, coeff_cov="uniform", coeff_mean="uniform",
            n_chunks=200)

    # Train cross-validation.
    lambda0s = np.linspace(0.2, 1000, 50)
    sigma0s = np.linspace(0.1, 100, 50)

    # Compute folds via kMeans.
    k = 10 # number of clusters.
    folds = kMeans_folds(k, data_coords)
    updatable_gp.train_cv_criterion(lambda0s, sigma0s, G, synth_data, data_std,
            criterion="k fold", folds=folds,
            out_path=os.path.join(results_folder, "./{}_fold_kMeans_residuals.pck".format(k)))


if __name__ == "__main__":
    main()
