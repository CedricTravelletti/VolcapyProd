import os
import numpy as np
import torch
from volcapy.update.universal_kriging import UpdatableGP
import volcapy.covariance.matern52 as kernel
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.lazy_updatable_covariance import UpdatableCovLazyTensor

import gpytorch
from ax import optimize


data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
# data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018/"

# Load some data.
grid = Grid.load(os.path.join(data_folder,
    "grid.pickle"))
volcano_coords = torch.from_numpy(
        grid.cells).float().detach()
G = torch.from_numpy(
        np.load(os.path.join(data_folder, "F_corrected_final.npy"))).float().detach()
data_values = torch.from_numpy(
        np.load(os.path.join(data_folder,"niklas_data_obs_corrected_final.npy"))).float()

# Define some GP model.
data_std = 0.1
sigma0 = 284.66
m0 = 2139.1
lambda0 = 651.58

updatable_gp = UpdatableGP(kernel, lambda0, torch.tensor([sigma0]), m0,
        volcano_coords, n_chunks=200)

print(updatable_gp.neg_log_likelihood(torch.Tensor([lambda0]),
        torch.Tensor([sigma0]), torch.Tensor([m0]), G, data_values))

"""
lazy_cov = UpdatableCovLazyTensor(updatable_gp.covariance)

logdet = gpytorch.logdet(lazy_cov)

inv_quad, logdet = gpytorch.inv_quad_logdet(cov_mat, inv_quad_rhs, logdet=True)

"""
best_parameters, best_values, experiment, model = optimize(
        parameters=[
          {
            "name": "m0",
            "type": "range",
            "bounds": [1000.0, 3000.0],
          },
          {
            "name": "sigma0",
            "type": "range",
            "bounds": [1.0, 500.0],
          },
          {
            "name": "lambda0",
            "type": "range",
            "bounds": [1.0, 2000.0],
          },
        ],
        # Booth function
        evaluation_function=lambda p: updatable_gp.neg_log_likelihood(
            torch.Tensor([p["lambda0"]]),
        torch.Tensor([p["sigma0"]]), torch.Tensor([p["m0"]]),
        G, data_values, data_std=0.1).item(),
        minimize=True,
    )
