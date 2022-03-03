import os
import torch
from volcapy.update.updatable_covariance import UpdatableGP
import volcapy.covariance.matern52 as kernel
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.lazy_updatable_covariance import UpdatableCovLazyTensor

from ax import optimize


# data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018/"

# Load some data.
grid = Grid.load(os.path.join(data_folder,
    "grid.pickle"))
volcano_coords = torch.from_numpy(
        grid.cells).float().detach()

# Define some GP model.
data_std = 0.1
sigma0 = 284.66
m0 = 2139.1
lambda0 = 651.58

updatable_gp = UpdatableGP(kernel, lambda0, torch.tensor([sigma0]), m0,
        volcano_coords, n_chunks=200)

lazy_cov = UpdatableCovLazyTensor(updatable_gp.covariance)

logdet = gpytorch.logdet(UpdatableCovLazyTensor)

"""
inv_quad, logdet = gpytorch.inv_quad_logdet(cov_mat, inv_quad_rhs, logdet=True)

best_parameters, best_values, experiment, model = optimize(
        parameters=[
          {
            "name": "x1",
            "type": "range",
            "bounds": [-10.0, 10.0],
          },
          {
            "name": "x2",
            "type": "range",
            "bounds": [-10.0, 10.0],
          },
        ],
        # Booth function
        evaluation_function=lambda p: (p["x1"] + 2*p["x2"] - 7)**2 + (2*p["x1"] + p["x2"] - 5)**2,
        minimize=True,
    )
"""
