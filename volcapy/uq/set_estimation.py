""" Set estimation functions, such as Vorob'ev expectation and so on.

"""
import numpy as np
import torch
from scipy.optimize import brentq


def vorobev_expectation_inds(coverage):
    """ Compute the Vorob'ev expectation of a random set, given the coverage
    function (excursion probabilities).

    Parameters
    ----------
    coverage: array(n_cells)
        Array containing the excursion probability for each cell.

    Returns
    -------
    excursion_inds: array
        Array of indices (in the cells list) that correspond to cells belonging
        to the Vorob'ev expectation.

    """
    # Find the Vorob'ev threshold.
    if torch.is_tensor(coverage):
        coverage = coverage.detach().float().numpy()

    vorobev_volume = np.sum(coverage)

    def f(x):
        return np.sum(coverage > x) - vorobev_volume

    vorobev_threshold = brentq(f, 0.0, 1.0)

    vorobev_excu_inds = coverage > vorobev_threshold

    return vorobev_excu_inds

def vorobev_quantile_inds(coverage, quantile):
    """ Compute the Vorob'ev expectation of a random set, given the coverage
    function (excursion probabilities).

    Parameters
    ----------
    coverage: array(n_cells)
        Array containing the excursion probability for each cell.

    Returns
    -------
    excursion_inds: array
        Array of indices (in the cells list) that correspond to cells belonging
        to the Vorob'ev expectation.

    """
    # Find the Vorob'ev threshold.
    if torch.is_tensor(coverage):
        coverage = coverage.detach().float().numpy()

    vorobev_excu_inds = coverage > quantile

    return vorobev_excu_inds
