""" Prepare ground truth for article experiments by samling from Niklas
posterior. We here use a small discretization.

AWS Version: we just update prior samples generated offline.
"""
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


def prepare_groundtruth(
        data_path, prior_sample_path,
        post_sample_path, post_data_sample_path):
    """ Given a realization from the prior, compute the corresponding
    conditional realization by updating.

    Parameters
    ----------
    data_path: string
        Path to the static data defining the situation (grid, forward, ...).
    prior_sample_path: string
        Path ta a realization from the prior.
    post_sample_path: float
        Where to save the computed posterior realization.
    post_data_sample_path: string
        Where to save the computed posterior realization of the data.

    """
    # Load
    F = torch.from_numpy(
            np.load(os.path.join(data_path, "F_niklas.npy"))).float().detach()
    F_full = torch.from_numpy(
            np.load(os.path.join(data_path, "F_full_surface.npy"))).float().detach()

    grid = Grid.load(os.path.join(data_path,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(
            grid.cells).float().detach()

    data_coords = torch.from_numpy(
            np.load(os.path.join(data_path, "niklas_data_coords.npy"))).float()
    data_coords_full = torch.from_numpy(
            np.load(os.path.join(data_path,"surface_data_coords.npy"))).float()
    data_values = torch.from_numpy(
            np.load(os.path.join(data_path,"niklas_data_obs.npy"))).float()

    print("Size of inversion grid: {} cells.".format(volcano_coords.shape[0]))
    print("Number of datapoints: {}.".format(data_coords.shape[0]))
    size = data_coords.shape[0]*volcano_coords.shape[0]*4 / 1e9
    cov_size = (volcano_coords.shape[0])**2 * 4 / 1e9
    print("Size of Covariance matrix: {} GB.".format(cov_size))
    print("Size of Pushforward matrix: {} GB.".format(size))

    prior_sample = torch.from_numpy(
            np.load(prior_sample_path)).float()

    # HYPERPARAMETERS, small volcano.
    data_std = 0.1
    sigma0 = 359.49
    m0 = -114.40
    lambda0 = 338.46

    myGP = InverseGaussianProcess(m0, sigma0, lambda0,
            volcano_coords, kernel,
            logger=logger)

    start = timer()

    post_sample = myGP.update_sample(prior_sample, F, data_values, data_std)

    end = timer()
    print("Sample updating run in {}s.".format(end - start))

    np.save(post_sample_path,
            post_sample.detach().cpu().numpy())

    post_data_sample = F_full @ post_sample
    np.save(post_data_sample_path,
            post_data_sample.detach().cpu().numpy())


if __name__ == "__main__":
    data_path = "/home/ubuntu/Dev/VolcapyProd/data/InversionDatas/stromboli_173018_cells/"
    prior_samples_folder = "/home/ubuntu/Dev/VolcapyProd/data/InversionDatas/stromboli_173018_cells/prior_samples/"
    post_samples_folder = "/home/ubuntu/Dev/VolcapyProd/data/InversionDatas/stromboli_173018_cells/post_samples/"
    post_data_samples_folder = "/home/ubuntu/Dev/VolcapyProd/data/InversionDatas/stromboli_173018_cells/post_data_samples/"

    for i in range(3, 300):
        print("Generating sample nr. {}.".format(i))
        prior_sample_path = os.path.join(
                prior_samples_folder, "prior_sample_{}.npy".format(i))
        post_sample_path = os.path.join(
                post_samples_folder, "post_sample_{}.npy".format(i))
        post_data_sample_path = os.path.join(
                post_data_samples_folder, "post_data_sample_{}.npy".format(i))

        prepare_groundtruth(data_path, prior_sample_path,
                post_sample_path, post_data_sample_path)
