""" Sample from a given posterior.

This module uses the residual kriging algorithm, hence it should be feeded with
samples from the prior.

"""
import os
import numpy as np
import torch
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess
from volcapy.grid.grid_from_dsm import Grid
import volcapy.covariance.matern32 as kernel

from timeit import default_timer as timer


def sample_posterior(gp, n_samples, prior_sample_folder, G, data_values, data_std, post_sample_folder):
    """ Given a realization from the prior, compute the corresponding
    conditional realization by updating.

    Parameters
    ----------
    static_data_folder: string
        Path to the static data defining the situation (grid, forward, ...).
    prior_sample_folder: string
        Path to a realization from the prior.
    post_sample_folder: float
        Where to save the computed posterior realization.

    """
    for i in range(100, 100 + n_samples):
        print("Generating sample nr. {}".format(i - 99))
        # The samples for reskrig are the one with nr. 100 and up.
        prior_sample_path = os.path.join(prior_sample_folder,
                "prior_sample_{}.npy".format(i))
        prior_sample = torch.from_numpy(np.load(prior_sample_path))

        start = timer()
        post_sample = gp.update_sample(prior_sample, G, data_values, data_std)
        end = timer()
        print("Sample updating run in {}s.".format(end - start))

        post_sample_path = os.path.join(
                post_sample_folder, "post_sample_{}.npy".format(i - 99))
        np.save(post_sample_path,
                post_sample.detach().cpu().numpy())

def sample_posterior_strategy(
        strategy_folder, sample_nr, prior_sample_folder, static_data_folder, n_samples):
    # Load static data.
    # Load static data.
    F = torch.from_numpy(
            np.load(os.path.join(static_data_folder, "F_full_surface.npy"))).float().detach()
    grid = Grid.load(os.path.join(static_data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()

    # Create GP (trained Matern 32).
    data_std = 0.1
    sigma0_matern32 = 527.84
    m0_matern32 = 549.15
    lambda0_matern32 = 891.66

    myGP = InverseGaussianProcess(m0_matern32, sigma0_matern32,
            lambda0_matern32,
            volcano_coords, kernel,
            n_chunks=200, n_flush=50)

    current_folder = os.path.join(strategy_folder, "sample_{}/".format(sample_nr))
    post_sample_folder = os.path.join(current_folder, "post_samples/")
    os.makedirs(post_sample_folder, exist_ok=True)

    # Load observed data.
    visited_inds = np.load(
                os.path.join(current_folder, "visited_inds.npy")).flatten()
    observed_data = torch.from_numpy(
            np.load(
                        os.path.join(current_folder,
                        "observed_data.npy")).flatten().reshape(-1, 1))
    G = F[visited_inds, :]

    sample_posterior(
            myGP, n_samples, prior_sample_folder,
            G, observed_data, data_std, post_sample_folder)


if __name__ == "__main__":
    sample_nr = 5


    static_data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
    # Samples for reskrig are in the same folder, but only use the samples from nr
    # 100 up.
    prior_sample_folder = "/storage/homefs/ct19x463/AISTATS_results/final_samples_matern32/"
    
    base_folder = "/storage/homefs/ct19x463/AISTATS_results/"
    strategy_folder = os.path.join(base_folder, "wIVR_final_small/")
    n_samples = 100

    sample_posterior_strategy(
            strategy_folder, sample_nr,
            prior_sample_folder, static_data_folder, n_samples)
