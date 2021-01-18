""" Prepare grid, data locations an forward for simple situations used for
diagnostics. The considered geometries are uniform cubic grids in one, two or
three dimensions.

"""
import numpy as np
from volcapy.grid.square_grid import Grid
from volcapy.forward.twodim_gravimetry import compute_forward


class ToyInverseProblem2d():
    """ Simple two dimensional inverse problem which can be used for
    diagnostics. The geometry a square [-1,1] x [-1, 1] with observations
    locations distributed on horizontal and vertical lines above/below
    left and right of the square.

    Parameters
    ----------
    grid: Grid
    data_coords_full: array
    G: array
    data_inds_up: arrray
    data_inds_down: array
    data_inds_left: array
    data_inds_right: array

    """
    def __init__(self, grid, data_coords_full, G,
            data_inds_up, data_inds_down, data_inds_left, data_inds_right):
        self.grid = grid
        self.data_coords_full = data_coords_full
        self.G = G
        self.data_inds_up = data_inds_up
        self.data_inds_down = data_inds_down
        self.data_inds_left = data_inds_left
        self.data_inds_right = data_inds_right

    @classmethod
    def build_problem(cls, n_cells_1d, n_data_1d,
            data_loc_offset, data_loc_length):
        """ Prepare data for a square geometry [-1,1] x [-1, 1] with observation
        locations on vertical and horizontal lines above, below, left and right of
        the square.
    
        Parameters
        ----------
        n_cells_1d: int
            Number of cells along one dimension of the grid.
            Total number of cells will be this number squared.
        n_data_1d: int
            Number of data points along one line. Since there are 4 lines along
            which the observations are distributed, total number of observation
            locations will be 4 times this.
        data_loc_offset: float
            Observations are distributed along straight horizontal and vertical
            line situaded at a fixed distance from the sides of the square. This
            defines their distance to the square.
        data_loc_length: float
            Length of each of the four lines where the observations will be
            distributed.
    
        """
        # Build the grid.
        grid = Grid(n_dims=2, n_cells_1d=n_cells_1d)
    
        # Create the data locations.
        data_coords_up = np.vstack(
                [np.linspace(-data_loc_length/2, data_loc_length/2,n_data_1d, endpoint=True),
                 np.repeat(1.0 + data_loc_offset, n_data_1d)]).T
    
        data_coords_down = np.vstack(
                [np.linspace(-data_loc_length/2, data_loc_length/2,n_data_1d, endpoint=True),
                 np.repeat(-1.0 - data_loc_offset, n_data_1d)]).T
    
        data_coords_left = np.vstack(
                [np.repeat(-1.0 - data_loc_offset, n_data_1d),
                 np.linspace(-data_loc_length/2,
                         data_loc_length/2,n_data_1d, endpoint=True)]).T
    
        data_coords_right = np.vstack(
                [np.repeat(1.0 + data_loc_offset, n_data_1d),
                 np.linspace(-data_loc_length/2,
                         data_loc_length/2,n_data_1d, endpoint=True)]).T
        
        data_coords_full = np.vstack([data_coords_up, data_coords_down,
                data_coords_left, data_coords_right])
    
        data_inds_up = list(range(0, n_data_1d))
        data_inds_down = list(range(n_data_1d, 2*n_data_1d))
        data_inds_left = list(range(2*n_data_1d, 3*n_data_1d))
        data_inds_right = list(range(3*n_data_1d, 4*n_data_1d))
    
        # Compute the forward.
        G = compute_forward(grid.cells, grid.res, grid.res, data_coords_full, n_procs=4)
        
        return ToyInverseProblem2d(grid, data_coords_full, G, 
                data_inds_up, data_inds_down, data_inds_left, data_inds_right)
