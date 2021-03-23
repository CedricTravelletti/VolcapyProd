""" Discrete Fourier transform operator in 2D.

"""
import numpy as np
from multiprocessing import Pool, RawArray
from volcapy.niklas.banerjee2d import banerjee


def compute_forward(coords, M, N, n_procs, forward_cutoff=None):
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
    forward_cutoff: int, optional
        If specified, only compute the forward for the coefficients up to
        the cutoff (ordered along square filling sequence).
        This is useful when building large-grid problems, where the full
        Fourier operator would be quadratic in the number of cells.

    Returns
    -------
    ndarray
        Forward operator, size n_data * n_cells.

    """
    M = int(np.sqrt(coords.shape[0]))
    n = coords.shape[0]

    if forward_cutoff is None:
        forward_cutoff = n

    G_re = np.zeros((forward_cutoff, n), dtype=np.float32)
    G_im = np.zeros((forward_cutoff, n), dtype=np.float32)

    Delta_x = np.max(coords[:, 0]) - np.min(coords[:, 0])
    Delta_y = np.max(coords[:, 1]) - np.min(coords[:, 1])
    delta_x = Delta_x / M
    scale_factor = Delta_x + delta_x

    # Generate the square filling sequence.
    fourier_inds = []
    for i in range(1, M + 1):
        for j in range(i):
            fourier_inds.append([j, i-1])
        for k in range(i-1):
            fourier_inds.append([i-1, k])


    print(forward_cutoff)
    for l in range(forward_cutoff):
        print(l)
        i, j = fourier_inds[l][0], fourier_inds[l][1]
        # ind = np.ravel_multi_index((i, j), (M, M))
        ind = l
        for k, cell in enumerate(coords):
            G_re[ind, k] = np.cos((2*np.pi / scale_factor) * (i * cell[0] +
                    j * cell[1]))
            G_im[ind, k] = np.sin((2*np.pi / scale_factor) * (i * cell[0] +
                    j * cell[1]))
    return G_re, G_im
