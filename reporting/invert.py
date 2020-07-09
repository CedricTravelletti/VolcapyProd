# File: forward_brute_force.py, Author: Cedric Travelletti, Date: 12.04.2019.
""" Run inversion. 
"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.inverse.gaussian_process import GaussianProcess
import volcapy.covariance.matern52 as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud, _array_to_vtk_point_cloud


import numpy as np
import os

# Now torch in da place.
import torch

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')

F = np.load("/home/cedric/PHD/Dev/VolcapySIAM/reporting/F_niklas.npy")
F = torch.as_tensor(F).float()
d_obs = np.load("/home/cedric/PHD/Dev/VolcapySIAM/reporting/niklas_data_obs.npy")
grid = Grid.load("/home/cedric/PHD/Dev/VolcapySIAM/reporting/grid.pickle")
cells_coords = grid.cells


# Careful: we have to make a column vector here.
data_std = 0.1
d_obs = torch.as_tensor(d_obs)[:, None]
n_data = d_obs.shape[0]
data_cov = torch.eye(n_data)
cells_coords = torch.as_tensor(cells_coords).detach()

# HYPERPARAMETERS
sigma0_init = 221.6
m0 = 2133.8
lambda0 = 462.0

# Define GP model.
myGP = GaussianProcess(F, d_obs, sigma0_init,
    data_std=data_std)

# Create the covariance pushforward.
cov_pushfwd = cl.compute_cov_pushforward(
        lambda0, F, cells_coords, cpu, n_chunks=20,
        n_flush=50)
K_d = torch.mm(F, cov_pushfwd)


m_post_d = myGP.condition_data(
        K_d, sigma0_init, concentrate=True)

# Compute diagonal of posterior covariance.
post_cov_diag = myGP.compute_post_cov_diag(
        cov_pushfwd, cells_coords, lambda0, sigma0_init, cl)

# Compute train_error
train_error = myGP.train_RMSE()
print("Train error (data conditioning): {}".format(train_error.item()))

# Compute LOOCV RMSE.
loocv_rmse = myGP.loo_error()
print("LOOCV error (data conditioning): {}".format(loocv_rmse.item()))

# Once finished, run a forward pass.
m_post_m, m_post_d = myGP.condition_model(
        cov_pushfwd, F, sigma0_init, concentrate=True)

# Compute train_error
train_error = myGP.train_RMSE()
print("Train error (model conditioning): {}".format(train_error.item()))

# Compute LOOCV RMSE.
loocv_rmse = myGP.loo_error()
print("LOOCV error (model conditioning): {}".format(loocv_rmse.item()))

irregular_array_to_point_cloud(cells_coords.numpy(), m_post_m.numpy(), "m_post.vtk")
_array_to_vtk_point_cloud(cells_coords.numpy(), m_post_m.numpy(), "m_post_irreg.vtk")
