""" Prepare grid, data locations an forward for a toy 2 dimensional inverse
problem where coefficients of the 2d DFT are observed.

"""
import numpy as np
import torch
from volcapy.grid.square_grid import Grid
from volcapy.forward.twodim_fourier import compute_forward


class ToyFourier2d():
    """ Simple two dimensional inverse problem which can be used for
    diagnostics. The geometry is a square [-1,1] x [-1, 1] with observations
    consisting of coefficients of the discrete Fourier transform (DFT) on the
    square.

    Parameters
    ----------
    grid: Grid
    G_re: array
    G_re: array

    """
    def __init__(self, grid, G_re, G_im, fourier_inds, n_cells_1d):
        self.grid = grid
        self.G_re = torch.from_numpy(G_re)
        self.G_im = torch.from_numpy(G_im)
        self.n_cells_1d = n_cells_1d
        self.fourier_inds = fourier_inds

        # Concatenate to make compatible with code for single observations.
        self.G = np.stack([G_re, G_im], axis=1)

        # Add point observations.
        self.G_pt = np.eye(self.G_re.shape[1])

    @classmethod
    def build_problem(cls, n_cells_1d, forward_cutoff=None):
        """ Prepare data for a square geometry [-1,1] x [-1, 1] with
        observations consisting of coefficients of the DFT.
    
        Parameters
        ----------
        n_cells_1d: int
            Number of cells along one dimension of the grid.
            Total number of cells will be this number squared.
        forward_cutoff: int, optional
            If specified, only compute the forward for the coefficients up to
            the cutoff (ordered along square filling sequence).
            This is useful when building large-grid problems, where the full
            Fourier operator would be quadratic in the number of cells.
    
        """
        # Build the grid.
        grid = Grid(n_dims=2, n_cells_1d=n_cells_1d)
    
        # Compute the forward.
        M = n_cells_1d
        N = n_cells_1d
        G_re, G_im, fourier_inds = compute_forward(grid.cells, M, N, n_procs=4,
                forward_cutoff=forward_cutoff)
        
        return cls(grid, G_re, G_im, fourier_inds, n_cells_1d)

    def index_to_1d(self, m, n):
        """ Converts a 2D index (in Fourier space) to a 1D index, which allows
        to retrieve the location inside the forward operator.

        """
        return np.ravel_multi_index((m, n), (self.n_cells_1d, self.n_cells_1d))
