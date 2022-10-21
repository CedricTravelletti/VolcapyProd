""" Compute predictions from trained GP models (Stromboli data).

"""
from volcapy.update.universal_kriging import UniversalUpdatableGP
import volcapy.covariance.matern32 as kernel
from volcapy.grid.grid_from_dsm import Grid
from volcapy.universal_kriging.basis_functions import *

import numpy as np
import os
import sys
import pickle

# Now torch in da place.
import torch

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')


# Lodad Niklas data.
data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
results_folder = "/storage/homefs/ct19x463/Dev/VolcapyProd/reporting/universal_kriging/results/Stromboli/"

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
# lambda0_cyl = 616.179487
# sigma0_cyl = 323.846154
lambda0_cyl, sigma0_cyl = 923.769231, 574.94

# Build trends: constant cylindrical.
x0 = volcano_coords[:, 0].mean() # Volcano center.
y0 = volcano_coords[:, 1].mean()
z0 = volcano_coords[:, 2].mean()

# Put through sigmoid filter.
dists_cyl = cylindrical(volcano_coords, x0, y0)
basis_fn_cyl = tanh_sigmoid(dists_cyl**2, saturation_length=2e6, inverted=True)

coeff_F_cyl = torch.hstack([
    torch.ones(volcano_coords.shape[0], 1),
    basis_fn_cyl.reshape(-1, 1)
]).float()

# Model with cylindrical trend.
updatable_gp_cyl = UniversalUpdatableGP(kernel, lambda0_cyl, sigma0_cyl,
        volcano_coords,
        coeff_F_cyl, coeff_cov='uniform', coeff_mean='uniform',
        n_chunks=200)

lambda0_cst = 846.871795
sigma0_cst = 467.333333
beta_hat_cst = torch.tensor([[544.8717]])

# Model with constant.
updatable_gp_cst = UniversalUpdatableGP(kernel, lambda0_cst, sigma0_cst,
        volcano_coords,
        torch.ones(volcano_coords.shape[0], 1).float(),
        coeff_cov='uniform', coeff_mean='uniform',
        n_chunks=200)

# Model with trend along N41 fault line.
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

updatable_gp_N41 = UniversalUpdatableGP(kernel, lambda0_N41, sigma0_N41,
        volcano_coords,
        coeff_F_N41,
        coeff_cov='uniform', coeff_mean='uniform',
        n_chunks=200)

# Compute posterior means.
post_mean_cyl = updatable_gp_cyl.predict_uniform(G, data_values, data_std)
post_mean_cst = updatable_gp_cst.predict_uniform(G, data_values, data_std)
post_mean_N41 = updatable_gp_N41.predict_uniform(G, data_values, data_std)

np.save(os.path.join(results_folder, "post_mean_cyl.npy"), post_mean_cyl.cpu().numpy())
np.save(os.path.join(results_folder, "post_mean_cst.npy"), post_mean_cst.cpu().numpy())
np.save(os.path.join(results_folder, "post_mean_N41.npy"), post_mean_N41.cpu().numpy())

# Compute cross-validation residuals.
residuals_loocv_cyl = updatable_gp_cyl.leave_1_out_residuals(G, data_values, data_std=0.1)
residuals_loocv_cst = updatable_gp_cst.leave_1_out_residuals(G, data_values, data_std=0.1)
residuals_loocv_N41 = updatable_gp_N41.leave_1_out_residuals(G, data_values, data_std=0.1)

np.save(os.path.join(results_folder, "residuals_loocv_cyl.npy"), residuals_loocv_cyl.cpu().numpy())
np.save(os.path.join(results_folder, "residuals_loocv_cst.npy"), residuals_loocv_cst.cpu().numpy())
np.save(os.path.join(results_folder, "residuals_loocv_N41.npy"), residuals_loocv_N41.cpu().numpy())

# Compute k-fold cross-validation for kMeans folds.
from volcapy.utils import kMeans_folds
k = 10 # number of clusters.
folds = kMeans_folds(k, data_coords)
residuals_kfolds_cyl = updatable_gp_cyl.k_fold_residuals(folds, G, data_values, data_std)
residuals_kfolds_cst = updatable_gp_cst.k_fold_residuals(folds, G, data_values, data_std)
residuals_kfolds_N41 = updatable_gp_N41.k_fold_residuals(folds, G, data_values, data_std)

np.save(os.path.join(results_folder, "residuals_kfolds_cyl.npy"), residuals_kfolds_cyl)
np.save(os.path.join(results_folder, "residuals_kfolds_cst.npy"), residuals_kfolds_cst)
np.save(os.path.join(results_folder, "residuals_kfolds_N41.npy"), residuals_kfolds_N41)

# Save folds indices.
with open(os.path.join(results_folder, "folds_inds.pkl"), "wb") as fp:
        pickle.dump(folds, fp)
