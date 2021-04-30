""" Set estimation functions, such as Vorob'ev expectation and so on.

"""
import numpy as np


def vorobev_expectation_inds(coverage):
    # Find the Vorob'ev threshold.
    coverage = coverage.detach().float().numpy()
    vorobev_volume = np.sum(coverage)
    def f(x):
        return np.sum(coverage > x) - vorobev_volume

    from scipy.optimize import brentq
    vorobev_threshold = brentq(f, 0.0, 1.0)
    print("Vorob'ev threshold: {}.".format(vorobev_threshold))

    # Plot estimated excursion set using coverage function.
    vorobev_excu_inds = coverage > vorobev_threshold

    print("Vorobe'v volume: {}.".format(vorobev_volume))
    print("Current estimate volume: {}.".format(np.sum(vorobev_excu_inds)))

    return vorobev_excu_inds
