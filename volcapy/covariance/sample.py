""" Interface to R for sampling from priors.

"""
import torch
import numpy as np
from rpy2.robjects import numpy2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# Activate auto-wrapping of numpy arrays as rpy2 objects.
numpy2ri.activate()

# Import the R RandomFields library.
rflib = importr("RandomFields")


def sample(kernel, sigma0, lambda0, m0, cells):
    """ Sample from a given covariance model.

    Parameters
    ----------
    kernel: One of the volcapy.covariance modules.
    sigma0: float
        Kernel standard deviation.
    lambda0: float
        Kernel lengthscale.
    m0: float
        Prior mean (constant model).
    cells: (n, d) ndarray or Tensor
        Coordinates of the points at which to sample.

    Returns
    -------
    sample: (n, 1) Tensor
        Sampled values.

    """
    if torch.is_tensor(cells):
        cells = cells.detach().cpu().numpy()
    if torch.is_tensor(sigma0):
        sigma0 = sigma0.detach().cpu().item()
    if torch.is_tensor(lambda0):
        lambda0 = lambda0.detach().cpu().item()

    scale = lambda0

    # Create the model.
    if kernel.KERNEL_FAMILY == "exponential":
        print("Sampling from exponential model")
        model = rflib.RMexp(var=sigma0**2,
                scale=lambda0)
    elif kernel.KERNEL_FAMILY == "squared_exponential":
        print("Sampling from squared exponential model")
        model = rflib.RMgauss(var=sigma0**2,
                scale=lambda0)
    elif kernel.KERNEL_FAMILY == "matern32":
        print("Sampling from Matern 3/2 model")
        model = rflib.RMmatern(nu=1.5, var=sigma0**2,
                scale=scale)
    elif kernel.KERNEL_FAMILY == "matern52":
        print("Sampling from Matern 5/2 model")
        model = rflib.RMmatern(nu=2.5, var=sigma0**2,
                scale=scale)
    else: raise ValueError("Kernel type not recognized.")

    # Sample.
    # Warning: the implementation seems to have changed. We have to make 
    # the model circulant now.
    simu = rflib.RFsimulate(rflib.RPtbm(model, linesimustep=10.0), cells)

    # Back to numpy, make column vector also.
    sample = torch.from_numpy(
            np.asarray(simu.slots['data']).astype(np.float32)[:, None])

    return sample + m0 * torch.ones(sample.shape)

def direct_sample(kernel, sigma0, lambda0, m0, cells):
    """ Sample directly (without turning bands or other approximations. 

    """
    n = cells.shape[0]
    cov_mat = sigma0**2 * kernel.compute_covariance(lambda0, cells)
    L = torch.cholesky(cov_mat).double()
    Z = torch.normal(mean=0, std=1.0, size=(n, 1)).double()
    return (m0 + L @ Z).float()
