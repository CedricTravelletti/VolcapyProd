""" Discrete Fourier transform operator in 2D.

"""
import numpy as np
from multiprocessing import Pool, RawArray
from volcapy.niklas.banerjee2d import banerjee


def compute_forward(coords):
    """ Compute the forward operator associated to a given topography/irregular
    grid. In the end, it only need a list of cells.

    Returns
    -------
    ndarray
        Forward operator, size n_data * n_cells.

    """
    M = coords.shape[0]
    G_re = np.zeros((M, M), dtype=np.float32)
    G_im = np.zeros((M, M), dtype=np.float32)

    for i in range(coords.shape[0]):
        for k, cell in enumerate(coords):
            G_re[i, k] = np.cos((2*np.pi / M) * i * k)
            G_im[i, k] = - np.sin((2*np.pi / M) * i * k)
    return G_re, G_im
