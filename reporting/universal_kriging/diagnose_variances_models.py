""" Compute the predictive variance for different folds, in order to diagnose if it is 
too small.

"""
from volcapy.update.updatable_covariance import UpdatableGP
from volcapy.update.universal_kriging import UniversalUpdatableGP
import volcapy.covariance.matern32 as kernel
from volcapy.grid.grid_from_dsm import Grid
from volcapy.universal_kriging.basis_functions import *
from volcapy.utils import kMeans_folds

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
results_folder = "/storage/homefs/ct19x463/Dev/VolcapyProd/reporting/universal_kriging/results/Stromboli/variance_diagnosis/"

G = torch.from_numpy(
        np.load(os.path.join(data_folder, "F_corrected_final.npy"))).float().detach()
grid = Grid.load(os.path.join(data_folder,
                "grid.pickle"))
volcano_coords = torch.from_numpy(grid.cells).float().detach()
data_coords = torch.from_numpy(
        np.load(os.path.join(data_folder,"niklas_data_coords_corrected_final.npy"))).float()
data_values = torch.from_numpy(
        np.load(os.path.join(data_folder,"niklas_data_obs_corrected_final.npy"))).float()


# We here use the non-universal constant GP from the volcano article.
data_std = 0.1
sigma0_matern32 = 284.66
m0_matern32 = 2139.1
lambda0_matern32 = 651.58

updatable_gp = UpdatableGP(kernel, lambda0_matern32, sigma0_matern32, m0_matern32, volcano_coords,
            n_chunks=200)

# Compute folds via kMeans.
k = 10 # number of clusters.
folds = kMeans_folds(k, data_coords)
# Save folds indices.
with open(os.path.join(results_folder, "folds_inds.pkl"), "wb") as fp:
        pickle.dump(folds, fp)

def partition_tensor(T, out_inds, axis):
    full_inds = np.array(list(range(T.shape[axis])))
    in_inds = np.delete(full_inds, out_inds)
    T_in = torch.index_select(T, axis, torch.LongTensor(in_inds))
    T_out = torch.index_select(T, axis, torch.LongTensor(out_inds))
    return T_in, T_out

"""
# For each fold, predict variances.
for i, fold_inds in enumerate(folds):
    # Partition the data.
    y_in, y_out = partition_tensor(data_values, fold_inds, axis=0)
    G_in, G_out = partition_tensor(G, fold_inds, axis=0)

    # Assimilate the data.
    updatable_gp.update(G_in, y_in, data_std)

    # Predict left out data.
    y_out_pred = G_out @ updatable_gp.mean_vec

    # Prediction covariance.
    data_cov = data_std**2 * torch.eye(y_out.shape[0]).float()
    pushfwd_out = updatable_gp.covariance.mul_right(G_out.t()).float().cpu()
    y_out_pred_cov = G_out @ pushfwd_out + data_cov

    # Save results.
    np.save(os.path.join(results_folder, "y_out_pred_fold_{}.npy".format(i)), y_out_pred.cpu().numpy())
    np.save(os.path.join(results_folder, "y_out_pred_cov_fold_{}.npy".format(i)), y_out_pred_cov.cpu().numpy())

    # Forget the data we have assimilated.
    updatable_gp.rewind(0)
"""

# Also compute posterior mean for non-universal model.
updatable_gp.update(G, data_values, data_std)
np.save(os.path.join(results_folder, "post_mean_nonuniv.npy"), updatable_gp.mean_vec.cpu().numpy())

# Now consider universal updatable GP and compute using fast-CV.
lambda0_cst = lambda0_matern32
sigma0_cst = sigma0_matern32
beta_hat_cst = torch.tensor([[544.8717]])

# Model with constant.
updatable_gp_cst = UniversalUpdatableGP(kernel, lambda0_cst, sigma0_cst,
        volcano_coords,
        torch.ones(volcano_coords.shape[0], 1).float(),
        coeff_cov='uniform', coeff_mean='uniform',
        n_chunks=200)

# Compute uinversal CV residuals manually.
for i, fold_inds in enumerate(folds):
    # Partition the data.
    y_in, y_out = partition_tensor(data_values, fold_inds, axis=0)
    G_in, G_out = partition_tensor(G, fold_inds, axis=0)

    # Assimilate the data.
    mean_vec = updatable_gp_cst.predict_uniform(G_in, y_in, data_std)

    # Predict left out data.
    y_out_pred = G_out @ mean_vec

    """
    # Prediction covariance.
    data_cov = data_std**2 * torch.eye(y_out.shape[0]).float()
    pushfwd_out = updatable_gp.covariance.mul_right(G_out.t()).float().cpu()
    y_out_pred_cov = G_out @ pushfwd_out + data_cov
    """

    # Save results.
    np.save(os.path.join(results_folder, "y_out_pred_univ_fold_{}.npy".format(i)), y_out_pred.cpu().numpy())
    # np.save(os.path.join(results_folder, "y_out_pred_cov_fold_{}.npy".format(i)), y_out_pred_cov.cpu().numpy())

    # Forget the data we have assimilated.
    updatable_gp_cst.rewind(0)

post_mean_cst = updatable_gp_cst.predict_uniform(G, data_values, data_std)
np.save(os.path.join(results_folder, "post_mean_cst.npy"), post_mean_cst.cpu().numpy())
print(updatable_gp_cst.beta_hat)

residuals_kfolds_cst = updatable_gp_cst.k_fold_residuals(folds, G, data_values, data_std=0.01)
k_tilde_inv_cst = updatable_gp_cst.K_tilde_inv

np.save(os.path.join(results_folder, "residuals_kfolds_cst_high_noise.npy"), residuals_kfolds_cst)
np.save(os.path.join(results_folder, "k_tilde_inv_cst.npy"), k_tilde_inv_cst)

updatable_gp_cst.predict_uniform(G, data_values, data_std)
print(updatable_gp_cst.beta_hat)
