#!/usr/bin/env python
# coding: utf-8

# In[1]:


""" Generate synthetic ground truth for the universal inversion paper. 
The ground truth generated here is sampled from a GP with fault line trend, 
using hyperparameters trained on Niklas data.

"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.update.universal_kriging import UniversalUpdatableGP
from volcapy.update.updatable_covariance import UpdatableGP
import volcapy.covariance.matern32 as kernel
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

data_folder = "/home/docker/data/InversionDatas/stromboli_173018"
save_folder = "/home/docker/data/samples"

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

# Define GP model (Niklas trained hyperparameters).
data_std = 0.1
sigma0 = 284.66
m0 = 2139.1
lambda0 = 651.58

# Build trends: constant + planar + cylindrical.
x0 = volcano_coords[:, 0].mean() # Volcano center.
y0 = volcano_coords[:, 1].mean()
z0 = volcano_coords[:, 2].mean()

coeff_mean = torch.tensor([m0, 0.0, 0.0]).reshape(-1, 1)
coeff_cov = torch.tensor([
            [200.0, 0, 0],
            [0, 0.05, 0],
            [0, 0, 0.05]
            ])
coeff_F = torch.hstack([
        torch.ones(volcano_coords.shape[0], 1),
        planar(
            volcano_coords,
            x0, y0, z0, phi=torch.tensor([45]), theta=torch.tensor([45])).reshape(-1, 1),
        cylindrical(
            volcano_coords,
            x0, y0).reshape(-1, 1)
        ])

# Model with trend.
updatable_gp = UniversalUpdatableGP(kernel, lambda0, torch.tensor([sigma0]),
            volcano_coords,
            coeff_F, coeff_cov, coeff_mean,
            n_chunks=200)


# In[ ]:


x0 = volcano_coords[:, 0].mean() # Volcano center.                             
y0 = volcano_coords[:, 1].mean()                                               
z0 = volcano_coords[:, 2].mean()                                               
      
# Build Fault Line trend.
origin_offset = 0                                                                                    
theta = 90 # equatorial plane.                                                 
phi = 135                                                                      
dists_N41 = planar(volcano_coords, x0, y0, z0, phi, theta,                     
            # cutoff_phi=45, cutoff_theta=90,                              
            # cutoff_x0=x0-300, cutoff_y0=y0-300, cutoff_z0=z0, fill_value=6000
            )                                                              
basis_fn_N41 = tanh_sigmoid(dists_N41, saturation_length=2500, inverted=True)
coeff_F = torch.hstack([                                                       
    torch.ones(volcano_coords.shape[0], 1),                                    
    basis_fn_N41.reshape(-1, 1)                                                
    ]).float()                                                                        


# # The first ground truth comes from a model with a planar and a cylindrical trend. Two models are competing: a well-specified universal krigin model and a model with a constant trend only.

# In[5]:


# Sample artificial volcano and 
# compute data at Niklas points.
ground_truth, true_trend_coeffs = updatable_gp.sample_prior()
noise = MultivariateNormal(loc=torch.zeros(n_data), covariance_matrix=data_std**2 * torch.eye(n_data)).rsample()
synth_data = G @ ground_truth + noise


np.save(os.path.join(save_folder, "ground_truth.npy"), ground_truth.cpu().numpy())
np.save(os.path.join(save_folder, "data_vals_niklas_points.npy"), synth_data.cpu().numpy())


"""
# In[3]:


# Plot ground truths.
from volcapy.plotting.plot_3D import get_plotly_surface, plot_surfaces, get_standard_slices
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}]],
    shared_yaxes=True,
    subplot_titles=("Trend", "Ground Truth", "Posterior Mean (universal)", "Posterior Mean (constant)"))

slice_z_gt, slice_y_gt, slice_x_gt = get_standard_slices(grid, ground_truth)
slice_z_gt_misspec, slice_y_gt_misspec, slice_x_gt_misspec = get_standard_slices(grid, ground_truth_misspec)

fig.add_trace(slice_x_gt)
fig.add_trace(slice_y_gt)
fig.add_trace(slice_z_gt)

cmap='jet'
vmin=1200
vmax = 2500
fig.update_layout(height=900, width=1200,
                  coloraxis=dict(colorscale=cmap,
                                 colorbar_thickness=25,
                                 colorbar_len=0.75,
                                 cmin=vmin, cmax=vmax))
fig.show()
"""
