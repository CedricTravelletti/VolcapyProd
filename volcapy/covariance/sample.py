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

    # Create the model.
    if kernel.KERNEL_FAMILY == "exponential":
        model = rflib.RMexp(var=sigma0**2,
                scale=lambda0)
    elif kernel.KERNEL_FAMILY == "squared_exponential":
        model = rflib.RMgauss(var=sigma0**2,
                scale=lambda0)
    elif kernel.KERNEL_FAMILY == "matern32":
        model = rflib.RMmatern(nu=1.5, var=sigma0**2,
                scale=lambda0)
    elif kernel.KERNEL_FAMILY == "matern52":
        model = rflib.RMmatern(nu=2.5, var=sigma0**2,
                scale=lambda0)
    else: raise ValueError("Kernel type not recognized.")

        # Sample.
    simu = rflib.RFsimulate(model, cells)

    # Back to numpy, make column vector also.
    sample = torch.from_numpy(
            np.asarray(simu.slots['data']).astype(np.float32)[:, None])

    return sample + m0 * torch.ones(sample.shape)
