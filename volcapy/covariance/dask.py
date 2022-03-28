""" Dask implementation of the covariance kernels.

"""
import dask.array as da
import dask_distance


def matern32(self, coords, lambda0):
    """ Matern 3/2 covariance kernel.

    Parameters
    ----------
    coords: (n_pts, n_dims) dask.array or Future
        Point coordinates.

    Returns
    -------
    covs: (n_pts, n_pts) delayed dask.array
        Pairwise covariance matrix.

    """
    dists = dask_distance.euclidean(coords)
    res = da.multiply(
            1 + (np.sqrt(3) / lambda0) * dists,
            da.exp(-(np.sqrt(3) / lambda0) * dists))
    return res
