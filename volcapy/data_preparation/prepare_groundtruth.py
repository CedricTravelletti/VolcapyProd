""" Prepare ground truth for article experiments by samling from Niklas
posterior. We here use a small discretization.
"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess
import volcapy.covariance.matern52 as kernel
from volcapy.grid.grid_from_dsm import Grid
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud
import sys

from timeit import default_timer as timer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import os

# Now torch in da place.
import torch

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')


def prepare_groundtruth(input_path):
    # Load
    F = torch.from_numpy(
            np.load(os.path.join(input_path, "F_niklas.npy"))).float().detach()
    F_full = torch.from_numpy(
            np.load(os.path.join(input_path, "F_full_surface.npy"))).float().detach()

    grid = Grid.load(os.path.join(input_path,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(
            grid.cells).float().detach()

    data_coords = torch.from_numpy(
            np.load(os.path.join(input_path, "niklas_data_coords.npy"))).float()
    data_coords_full = torch.from_numpy(
            np.load(os.path.join(input_path,"surface_data_coords.npy"))).float()
    data_values = torch.from_numpy(
            np.load(os.path.join(input_path,"niklas_data_obs.npy"))).float()

    print("Size of inversion grid: {} cells.".format(volcano_coords.shape[0]))
    print("Number of datapoints: {}.".format(data_coords.shape[0]))
    size = data_coords.shape[0]*volcano_coords.shape[0]*4 / 1e9
    cov_size = (volcano_coords.shape[0])**2 * 4 / 1e9
    print("Size of Covariance matrix: {} GB.".format(cov_size))
    print("Size of Pushforward matrix: {} GB.".format(size))

    # HYPERPARAMETERS, small volcano.
    data_std = 0.1
    sigma0 = 359.49
    m0 = -114.40
    lambda0 = 338.46

    myGP = InverseGaussianProcess(m0, sigma0, lambda0,
            volcano_coords, kernel,
            logger=logger)

    for i in range(n_realizations):
        start = timer()
    
        post_sample = myGP.sample_posterior(F, data_values, data_std)
    
        end = timer()
        print("Posterior sampling run in {}s.".format(end - start))
    
        """
        irregular_array_to_point_cloud(volcano_coords.numpy(),
                post_sample.detach().cpu().numpy(),
                os.path.join(input_path, "post_sample.vtk"), fill_nan_val=-20000.0)
    
        prior_sample = myGP.sample_prior()
        irregular_array_to_point_cloud(volcano_coords.numpy(),
                prior_sample.detach().cpu().numpy(),
                os.path.join(input_path, "prior_sample.vtk"), fill_nan_val=-20000.0)
    
        """
        np.save(os.path.join(input_path, "post_sample_{}.npy".format(i)),
                post_sample.detach().cpu().numpy())
        """
        np.save(os.path.join(input_path, "prior_sample.npy"),
                prior_sample.detach().cpu().numpy())
        """
        post_data_sample = F_full @ post_sample
        np.save(os.path.join(input_path, "post_data_sample_{}.npy".format(i)),
                post_data_sample.detach().cpu().numpy())


if __name__ == "__main__":
    print("Usage: prepare_groundtruth.py input_path n_realizations")
    input_path = sys.argv[1]
    n_realizations = int(sys.argv[2])
    prepare_groundtruth(input_path)
