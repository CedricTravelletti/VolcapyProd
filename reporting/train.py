""" Train hyperparameters.
"""
from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess
import volcapy.covariance.matern52 as kernel
from volcapy.grid.grid_from_dsm import Grid

from timeit import default_timer as timer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import os
import sys

# Now torch in da place.
import torch

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')


def main(data_folder):
    # Load
    F = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_niklas.npy"))).float().detach()

    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(
            grid.cells).float().detach()

    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"niklas_data_coords.npy"))).float()
    data_values = torch.from_numpy(
            np.load(os.path.join(data_folder,"niklas_data_obs.npy"))).float()

    print("Size of inversion grid: {} cells.".format(volcano_coords.shape[0]))
    print("Number of datapoints: {}.".format(data_coords.shape[0]))
    size = data_coords.shape[0]*volcano_coords.shape[0]*4 / 1e9
    cov_size = (volcano_coords.shape[0])**2 * 4 / 1e9
    print("Size of Covariance matrix: {} GB.".format(cov_size))
    print("Size of Pushforward matrix: {} GB.".format(size))

    # HYPERPARAMETERS
    data_std = 0.1
    sigma0 = 100.6
    m0 = 1000.0
    lambda0 = 300.0

    start = timer()
    myGP = InverseGaussianProcess(m0, sigma0, lambda0,
            volcano_coords, kernel,
            logger=logger)

    """
    m_post_m, m_post_d = myGP.condition_model(F, data_values, data_std,
            concentrate=False,
            is_precomp_pushfwd=False)
    """

    end = timer()
    print("Non-sequential inversion run in {} s.".format(end - start))

    # Train.
    myGP.train(np.linspace(20, 1400, 14), F, data_values, data_std,
            out_path="./train_res.pck",
            n_chunks=100,
            n_flush=1)


if __name__ == "__main__":
    data_folder = sys.argv[1]
    main(data_folder)
