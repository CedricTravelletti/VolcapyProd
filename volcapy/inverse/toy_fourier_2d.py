""" Prepare grid, data locations an forward for a toy 2 dimensional inverse
problem where coefficients of the 2d DFT are observed.

"""
import numpy as np
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
    def __init__(self, grid, G_re, G_im, n_cells_1d):
        self.grid = grid
        self.G_re = G_re
        self.G_im = G_im
        self.n_cells_1d = n_cells_1d

    @classmethod
    def build_problem(cls, n_cells_1d):
        """ Prepare data for a square geometry [-1,1] x [-1, 1] with
        observations consisting of coefficients of the DFT.
    
        Parameters
        ----------
        n_cells_1d: int
            Number of cells along one dimension of the grid.
            Total number of cells will be this number squared.
    
        """
        # Build the grid.
        grid = Grid(n_dims=2, n_cells_1d=n_cells_1d)
    
        # Compute the forward.
        M = n_cells_1d
        N = n_cells_1d
        G_re, G_im = compute_forward(grid.cells, M, N, n_procs=4)
        
        return cls(grid, G_re, G_im, n_cells_1d)

    def index_to_1d(self, m, n):
        """ Converts a 2D index (in Fourier space) to a 1D index, which allows
        to retrieve the location inside the forward operator.

        """
        return np.ravel_multi_index((m, n), (self.n_cells_1d, self.n_cells_1d))
