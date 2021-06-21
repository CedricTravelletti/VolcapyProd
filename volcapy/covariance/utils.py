""" Utilities for the covariance submodule.

"""

# Maximal lenght to consider in the root finding.
MAX_DIST = 100000.0


def compute_practical_range(kernel, lambda0):
    """ Get the practical range associated to some lengthscale for a given
    kernel.

    The practical range is defined as the distance at which the correlation
    decreases to 5% of its value at 0.

    Parameters
    ----------
    kernel: volcapy.covariance submoduel
    lambda0: float
        Lengthscale parameter.

    Returns
    -------
    float
        Practical range.

    """
    from scipy.optimize import brentq

    def f(x):
        return kernel.compute_cov_dist(x, lambda0) - 0.05
    practical_range = brentq(f, 0.0, MAX_DIST)
    return practical_range
