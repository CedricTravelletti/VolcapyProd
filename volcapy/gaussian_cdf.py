""" Compute coverage function, excursion sets and related quantities.

"""
import torch
from mvnorm import multivariate_normal_cdf


def gaussian_cdf(mean_vec, cov_mat, lower, upper=None):
    """ Computes coverage function at a fixed set of location.

    Ths function should be used when a mean vector and covariance matrix for a
    set of locations have been precomputed.

    Uses the torch implementation of the multivariate CDF by Sebastien Marmin,
    see torch-mvn.

    Parameters
    ----------
    mean_vec: (M, p) Tensor
        Mean vector at each of the M locations. mean_vec[i, :] should give the
        mean vector at location nr i.
    cov_mat: (M, p, p) Tensor
        Covariance matrix at each of the M locations. cov_mat[i, :, :] should
        give the covariance matrix at location nr i.
    lower: (p) Tensor
        List of lower threshold for each response. The excursion set is the set
        where responses are above the specified threshold.
        Note that np.inf is supported. WARNING: wrong shapes can cause
        unexpected results.
    upper: (p) Tensor
        List of upper threshold for each response. The excursion set is the set
        where responses are above the specified threshold.
        If not provided, defaults to + infinity.

    Returns
    -------
    excursion_prob: (M) Tensor
        Probability to be in excursion at each location.

    """
    # Might want to make it the same dimension as the number of points.
    # Works this way, but might be clearer.
    lower = torch.stack(cov_mat.shape[0] * [lower], dim=0)
    if upper is not None: upper = torch.stack(cov_mat.shape[0] * [upper], dim=0)

    """
    id = torch.eye(cov_mat.shape[2])
    jitter = 1e-1 * id.repeat(cov_mat.shape[0], *(id.dim() * [1]))
    """
    # print(jitter)
    cdf = multivariate_normal_cdf(
            lower=lower, upper=upper,
            loc=mean_vec, covariance_matrix=cov_mat)
    return cdf
