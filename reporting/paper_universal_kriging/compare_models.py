""" Compare different trend models with different CV criterion. 

"""
import numpy as np
import os
import sys

from volcapy.update.universal_kriging import UniversalUpdatableGP
import volcapy.covariance.matern32 as kernel
from volcapy.grid.grid_from_dsm import Grid
from volcapy.universal_kriging.basis_functions import *
from volcapy.utils import kMeans_folds

import torch
torch.set_num_threads(8)


base_folder = "/home/cedric/PHD/Dev/VolcapyProd/data"
base_folder = sys.argv[1]
data_folder = os.path.join(base_folder, "InversionDatas/stromboli_173018")

# Load the data.
grid = Grid.load(os.path.join(data_folder,
                "grid.pickle"))
volcano_coords = torch.from_numpy(grid.cells).float().detach()
data_coords = torch.from_numpy(np.load(os.path.join(data_folder, "niklas_data_coords.npy"))).float().detach()
G = torch.from_numpy(
        np.load(os.path.join(data_folder, "F_corrected_final.npy"))).float().detach()
data = torch.from_numpy(
        np.load(os.path.join(data_folder,"niklas_data_obs_corrected_final.npy"))).float()
data_std = 0.1

# Model with trend along N41 fault line.
x0 = volcano_coords[:, 0].mean() # Volcano center.
y0 = volcano_coords[:, 1].mean()
z0 = volcano_coords[:, 2].mean()

lambda0_N41, sigma0_N41 = 1231.35, 790.17

origin_offset = 0
theta = 90 # equatorial plane.
phi = 135

dists_N41 = planar(volcano_coords, x0, y0, z0, phi, theta,
                # cutoff_phi=45, cutoff_theta=90,
                # cutoff_x0=x0-300, cutoff_y0=y0-300, cutoff_z0=z0, fill_value=6000
                )
basis_fn_N41 = tanh_sigmoid(dists_N41, saturation_length=2500, inverted=True)

coeff_F_N41 = torch.hstack([
    torch.ones(volcano_coords.shape[0], 1),
    basis_fn_N41.reshape(-1, 1)
    ]).float()

model = UniversalUpdatableGP(kernel, lambda0_N41, sigma0_N41,
        volcano_coords,
        coeff_F_N41,
        coeff_cov='uniform', coeff_mean='uniform',
        n_chunks=200)


def compute_criterions(model):
    # Compute posterior means.
    post_mean = model.predict_uniform(G, data, data_std)

    rmse = model.prediction_rmse(G, data)
    nll = model.nll(G, data)
    # bic = model.bayes_information_criterion(G, data)
    
    # Compute cross-validation residuals.
    residuals_loocv = model.leave_1_out_residuals(G, data, data_std=0.1)
    residuals_loocv_decorr = model.decorrelated_leave_1_out_residuals(G, data, data_std=0.1)
    
    folds = kMeans_folds(2, data_coords)
    residuals_2folds_clustered = model.k_fold_residuals(folds, G, data, data_std)
    residuals_2folds_clustered_decorr = model.decorrelated_k_fold_residuals(folds, G, data, data_std)
    
    folds = kMeans_folds(10, data_coords)
    residuals_10folds_clustered = model.k_fold_residuals(folds, G, data, data_std)
    residuals_10folds_clustered_decorr = model.decorrelated_k_fold_residuals(folds, G, data, data_std)

    return (rmse, nll, bic,
            residuals_loocv, residuals_loocv_decorr,
            residuals_2folds_clustered, residuals_2folds_clustered_decorr,
            residuals_10folds_clustered, residuals_10folds_clustered_decorr)

compute_criterions(model)
