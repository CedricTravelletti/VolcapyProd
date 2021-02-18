""" Various utility functions.

"""
import numpy as np


def _make_column_vector(y):
    """ Make sure the data is a column vector.

    """
    if ((len(y.shape) >= 2) and (y.shape[1] == 1)):
        return y
    elif len(y.shape) == 1:
        return y.reshape(-1, 1)
    else:
        raise ValueError("Shape of data vector {} is not valid. Please provide a column vector.".format(y.shape))

# Generate R-sequences in arbitrary dimensions.
# Those are space filling sequences.
# Based on
# https://stats.stackexchange.com/questions/25528/do-low-discrepancy-sequences-work-in-discrete-spaces
# and
# http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/  


# Use Newton-Rhapson-Method to compute the plastic constant.
def gamma(d):
    x=1.0000
    for i in range(20):
        x = x-(pow(x,d+1)-x-1)/((d+1)*pow(x,d)-1)
    return x


def r_sequence(n_samples, dim, n_cells_1d):
    """ Generate a space filling sequence in a discretized hypercube.

    Parameters
    ----------
    n_samples: int
        How many samples to generate.
    dim: int
        Dimension of the hypercube.
    n_cells_1d: int
        Side lenghts of the hypercube.

    Returns
    -------
    samples: (n_samples, dim) array

    """
    g = gamma(dim)
    alpha = np.zeros(dim)                 
    for j in range(dim):
        alpha[j] = pow(1/g,j+1) %1
    z = np.zeros((n_samples, dim))    
    c = np.zeros((n_samples,dim)).astype(int)  
    
    for i in range(n_samples):
        z[i, :] = (0.5 + alpha*(i+1)) %1
        c[i, :] = (np.floor(n_cells_1d *z[i, :])).astype(int)
    return c
