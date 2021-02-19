""" Discrete Fourier transform operator in 2D.

"""
import numpy as np
from multiprocessing import Pool, RawArray
from volcapy.niklas.banerjee2d import banerjee


def compute_forward(coords, M, N, n_procs):
    """ Compute the forward operator associated to a given topography/irregular
    grid. In the end, it only need a list of cells.

    Parameters
    ----------
    coords: (n_cell, n_dims) ndarray
        Cells centroid coordinates.
    M: int
        Number of cells along x-dimension.
    N: int
        Number of cells along y-dimension.
    n_procs: int
        Number of processes to use to parallelize computation.

    Returns
    -------
    ndarray
        Forward operator, size n_data * n_cells.

    """
    M = int(np.sqrt(coords.shape[0]))
    n = coords.shape[0]
    G_re = np.zeros((n, n), dtype=np.float32)
    G_im = np.zeros((n, n), dtype=np.float32)

    Delta_x = np.max(coords[:, 0]) - np.min(coords[:, 0])
    Delta_y = np.max(coords[:, 1]) - np.min(coords[:, 1])
    delta_x = Delta_x / M
    scale_factor = Delta_x + delta_x

    for i in range(M):
        for j in range(M):
            for k, cell in enumerate(coords):
                ind = np.ravel_multi_index((i, j), (M, M))
                G_re[ind, k] = np.cos((2*np.pi / scale_factor) * (i * cell[0] +
                        j * cell[1]))
                G_im[ind, k] = np.sin((2*np.pi / scale_factor) * (i * cell[0] +
                        j * cell[1]))
    return G_re, G_im
