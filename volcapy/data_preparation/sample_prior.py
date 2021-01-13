""" Generate realizations from the prior.

"""
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess
import volcapy.covariance.matern52 as kernel
from volcapy.grid.grid_from_dsm import Grid
import sys

from timeit import default_timer as timer

import numpy as np
import os
import torch


def sample_prior(input_path, n_realizations):
    output_path = os.path.join(input_path, "prior_samples_v2/")
    os.makedirs(output_path, exist_ok=True)

    # Load
    grid = Grid.load(os.path.join(input_path,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(
            grid.cells).float().detach()

    # HYPERPARAMETERS.
    data_std = 0.1
    sigma0 = 359.49
    m0 = -114.40
    lambda0 = 338.46

    myGP = InverseGaussianProcess(m0, sigma0, lambda0,
            volcano_coords, kernel,
            n_chunks=70, n_flush=50)

    # TODO: Change. Here we had already produced 3 samples.
    for i in range(1, n_realizations):
        start = timer()
        prior_sample = myGP.sample_prior()
        end = timer()

        print("Prior sampling run in {}s.".format(end - start))
        np.save(os.path.join(output_path, "prior_sample_{}.npy".format(i)),
                prior_sample.detach().cpu().numpy())


if __name__ == "__main__":
    print("Usage: sample_prior.py input_path n_realizations")
    input_path = sys.argv[1]
    n_realizations = int(sys.argv[2])
    sample_prior(input_path, n_realizations)
