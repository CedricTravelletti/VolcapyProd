""" Sample from a given posterior.

This module uses the residual kriging algorithm, hence it should be feeded with
samples from the prior.

"""
import os
import numpy as np


# Samples for reskrig are in the same folder, but only use the samples from nr
# 100 up.
prior_sample_folder = "/storage/homefs/ct19x463/AISTATS_results/final_samples_matern32/"


def sample_posterior(gp, n_samples, prior_sample_folder, G, data_values, post_sample_folder):
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
    # Load
    for i in range(100, 100 n_samples):
        # The samples for reskrig are the one with nr. 100 and up.
        prior_sample_path = os.path.join(prior_sample_folder,
                "prior_sample_{}.npy".format(i))
        prior_sample = torch.from_numpy(np.load(prior_sample_path))

        start = timer()
        post_sample = myGP.update_sample(prior_sample, G, data_values, data_std)
        end = timer()
        print("Sample updating run in {}s.".format(end - start))

        post_sample_path = os.path.join(
                post_sample_folder, "post_sample_{}.npy")
        np.save(post_sample_path,
                post_sample.detach().cpu().numpy())
