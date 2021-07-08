""" Generate realizations from the prior.

"""
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess
import volcapy.covariance.matern32 as kernel
from volcapy.grid.grid_from_dsm import Grid
import sys

from timeit import default_timer as timer

import numpy as np
import os
import torch


def sample_prior(input_path, output_path, n_realizations):
    os.makedirs(output_path, exist_ok=True)

    # Load
    grid = Grid.load(os.path.join(input_path,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(
            grid.cells).float().detach()

    # HYPERPARAMETERS.
    data_std = 0.1

    sigma0_exp = 308.89 # WARNING: Old values.
    m0_exp = 535.39
    lambda0_exp = 1925.0

    sigma0_matern32 = 284.66
    m0_matern32 = 2139.1
    lambda0_matern32 = 651.58

    sigma0_matern52 = 258.40
    m0_matern52 = 2120.93
    lambda0_matern52 = 441.05

    myGP = InverseGaussianProcess(m0_matern32, sigma0_matern32,
            lambda0_matern32,
            volcano_coords, kernel,
            n_chunks=70, n_flush=50)

    for i in range(500, 500 + n_realizations):
        start = timer()
        prior_sample = myGP.sample_prior()
        end = timer()

        print("Prior sampling run in {}s.".format(end - start))
        np.save(os.path.join(output_path, "prior_sample_{}.npy".format(i)),
                prior_sample.detach().cpu().numpy())


if __name__ == "__main__":
    print("Usage: sample_prior.py input_path output_path n_realizations")
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    n_realizations = int(sys.argv[3])
    sample_prior(input_path, output_path, n_realizations)
