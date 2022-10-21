""" Collection of basis functions for universal kriging.

"""
import numpy as np
import torch


def cylindrical(coords, x0, y0):
    """ Cylindrical basis function: radial distance from center (x0, y0)
    
    """
    return torch.sqrt((coords[:, 0] - x0)**2 + (coords[:, 1] - y0)**2)

def planar(coords, x0, y0, z0, phi, theta,
        cutoff_x0=None, cutoff_y0=None, cutoff_z0=None, cutoff_phi=None, cutoff_theta=None, fill_value=None):
    """ Planar basis function: distance from a plane throught (x0, y0, z0) with 
    normal vector given by theta and phi in spherical coordinates.
    
    """
    phi, theta = torch.tensor([np.radians(phi)]), torch.tensor([np.radians(theta)])
    n = torch.tensor([
        [torch.cos(phi) * torch.sin(theta)],
        [torch.sin(phi) * torch.sin(theta)],
        [torch.cos(theta)]])

    # Compute distance to plane.
    dists = torch.abs(n[0] * (coords[:, 0] - x0) + n[1] * (coords[:, 1] - y0) + n[2] * (coords[:, 2] - z0))

    # TODO: The below is used to cut the fault line. 
    # The cut is done according to the angle with a given cutoff vector. 
    # As soon as the angle exceeds 90 degrees, values are set to zero.
    if (cutoff_theta is not None and cutoff_phi is not None and fill_value is not None):
        cutoff_phi, cutoff_theta = torch.tensor([np.radians(cutoff_phi)]), torch.tensor([np.radians(cutoff_theta)])
        n_cutoff = torch.tensor([
            [torch.cos(cutoff_phi) * torch.sin(cutoff_theta)],
            [torch.sin(cutoff_phi) * torch.sin(cutoff_theta)],
            [torch.cos(cutoff_theta)]])
        # Compute sign of scalar product, to see on which side the points lie.
        mask = (n_cutoff[0] * (coords[:, 0] - cutoff_x0) + n_cutoff[1] * (coords[:, 1] - cutoff_y0) + n_cutoff[2] * (coords[:, 2] - cutoff_z0) < 0)
        dists[mask] = fill_value
    return dists 

def tanh_sigmoid(dists, saturation_length, inverted=False):
    """ Rescales the computed distances (planar, cylindrical, ...) by a tanh sigmoid function.

    Parameters
    ----------
    dists: torch.Tensor [n_pts]
        Distances compute by another basis function (planar, cylindrical, ...).
    saturation_length: float
        Length at which to saturate the tanh (reaches 1).
    inverted: bool, defaults to False
        If True, then return a decreasing function, 1 - tanh(dists).

    Returns
    -------
    vals: torch.Tensor [n_pts]

    """
    saturation_length = torch.tensor([saturation_length])
    if inverted is False:
         return torch.tanh(dists * (0.5 * np.pi / saturation_length))
    elif inverted is True:
         return 1 - torch.tanh(dists * (0.5 * np.pi / saturation_length))

def _haar(x):
    if x < 0: return 0.0
    elif x < 0.5: return 1.0
    elif x < 1: return -1.0
    else: return 0.0
vhaar = np.vectorize(_haar)

def haar(x, n, k):
    return 2.0**(n/2.0) * vhaar(2.0**n * x - k)

def build_tensor_haar_basis(coords, nmin=-2, nmax=1, kmin=-2, kmax=2):
    """ Build the design matrix for Haar wavelet basis in 3D 
    via tensor products.

    The full basis has n and k in Z. Here the user specifies the minimum 
    and maximum values for the basis indices.

    """
    x0 = coords[:, 0].min()
    y0 = coords[:, 1].min()
    z0 = coords[:, 2].min()
    
    # Rescale to unit interval so we do not wast time.
    x, y, z = coords[:, 0] - x0, coords[:, 1] - y0, coords[:, 2] - z0
    x = x / (x.max() - x.min())
    y = y / (y.max() - y.min())
    z = z / (z.max() - z.min())
    
    ns = np.arange(nmin, nmax, 1)
    ks = np.arange(kmin, kmax, 1)
    stacked_funs = np.zeros((x.shape[0], 0))
    for n1 in ns:
        for k1 in ks:
            for n2 in ns:
                for k2 in ks:
                    for n3 in ns:
                        for k3 in ks:
                            stacked_funs = np.hstack(
                                    [stacked_funs,
                                        (haar(x - x0, n1, k1)
                                            * haar(y - y0, n2, k2)
                                            * haar(z - z0, n3, k3)).reshape(-1, 1)
                                    ])
    return torch.from_numpy(stacked_funs).float()
