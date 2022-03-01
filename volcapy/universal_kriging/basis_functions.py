""" Collection of basis functions for universal kriging.

"""
import numpy as np
import torch


def cylindrical(coords, x0, y0):
    """ Cylindrical basis function: radial distance from center (x0, y0)
    
    """
    return torch.sqrt((coords[:, 0] - x0)**2 + (coords[:, 1] - y0)**2)

def planar(coords, x0, y0, z0, phi, theta):
    """ Planar basis function: distance from a plane throught (x0, y0, z0) with 
    normal vector given by theta and phi in spherical coordinates.
    
    """
    phi, theta = torch.tensor([np.radians(phi)]), torch.tensor([np.radians(theta)])
    n = torch.tensor([
        [torch.cos(phi) * torch.sin(theta)],
        [torch.sin(phi) * torch.sin(theta)],
        [torch.cos(theta)]])
    return torch.abs(n[0] * (coords[:, 0] - x0) + n[1] * (coords[:, 1] - y0) + n[2] * (coords[:, 2] - z0))

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
