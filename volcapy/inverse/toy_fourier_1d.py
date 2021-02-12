""" Prepare grid, data locations an forward for a toy 2 dimensional inverse
problem where coefficients of the 2d DFT are observed.

"""
import numpy as np
from volcapy.grid.square_grid import Grid
from volcapy.forward.onedim_fourier import compute_forward


class ToyFourier1d():
    """ Simple oine dimensional inverse problem which can be used for
    diagnostics. The geometry is a line [-1,1], with observations
    consisting of coefficients of the discrete Fourier transform (DFT) on the
    square.

    Parameters
    ----------
    grid: Grid
    G_re: array
    G_re: array

    """
    def __init__(self, grid, G_re, G_im):
        self.grid = grid
        self.G_re = G_re
        self.G_im = G_im

    @classmethod
    def build_problem(cls, n_cells_1d):
        """ Prepare data for a line geometry [-1,1], with
        observations consisting of coefficients of the DFT.
    
        Parameters
        ----------
        n_cells_1d: int
            Number of cells along one dimension of the grid.
            Total number of cells will be this number squared.
    
        """
        # Build the grid.
        grid = Grid(n_dims=1, n_cells_1d=n_cells_1d)
    
        # Compute the forward.
        G_re, G_im = compute_forward(grid.cells)
        
        return cls(grid, G_re, G_im)
