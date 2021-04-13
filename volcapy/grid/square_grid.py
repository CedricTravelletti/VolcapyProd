""" Simple regular uniform grids in any dimension.
This is used for generating synthetic datasets that can be used for
diagnostics.

The domains we generate here are hypercubes (in any dimension) over the
interval [-1. 1]. I.e. the domain is [-1, 1]^(n_dims).

This domain is then discretized into uniform cubic cells. The user can specify
the number of cells along one axis via *n_cells_1d*. Then, the total number of
cells will be n_cells_1d^(n_dims).

"""
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt


class Grid():
    def __init__(self, n_dims, n_cells_1d):
        self.n_dims = n_dims
        self.res = 2.0 / (n_cells_1d - 1)
        self.n_cells_1d = n_cells_1d
        n_cells = n_cells_1d**n_dims

        if n_dims == 1:
            self.x = np.mgrid[-1.0:1.0:n_cells_1d*(1j)]
            self.cells = self.x[:, None]
        elif n_dims == 2:
            self.x, self.y = np.mgrid[-1.0:1.0:n_cells_1d*(1j),
                    -1.0:1.0:n_cells_1d*(1j)]
            self.cells = np.vstack([self.x.ravel(), self.y.ravel()]).T

        """
        coords_1d = np.arange(-1 + self.res / 2, 1, self.res).reshape(-1, 1)
        cells = coords_1d

        for i in range(1, n_dims):
            # Stack vertically to the size of the full cells array.
            # Note that at dimension i, the previous cell array has size n_cells^(i-1).
            new_dim_1d_coords = np.vstack(n_cells_1d**(i - 1) * [coords_1d])

            # Now concatenate with all permuations.
            old_cells = cells
            cells = np.hstack([old_cells, new_dim_1d_coords])
            for j in range(1, n_cells_1d):
                cells = np.vstack(
                        [cells, np.hstack([old_cells, np.roll(new_dim_1d_coords, j)])])
        """

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
                t = pickle.load(f)
                return cls(
                        t['cells'], t['res'])

    def save(self, path):
        pickled_grid = {
                'cells': self.cells, 'res': self.res}

        with open(path, 'wb') as f:
                pickle.dump(pickled_grid, f)

    def __getitem__(self, index):
        return self.cells[index]

    @property
    def shape(self):
        return self.cells.shape

    def plot_values(self, vals, cmap=None, vmin=None, vmax=None, colorbar=None, outfile=None):
        """ Plot a dataset over the grid. The values should be an array
        defininf the value of the data at each grid point.

        Parameters
        ----------
        vals: (n_cells) array

        """
        if self.n_dims == 2:
            # Generate regular gridding over which to plot.
            min_x = np.min(self.cells[:, 0])
            max_x = np.max(self.cells[:, 0])
            min_y = np.min(self.cells[:, 1])
            max_y = np.max(self.cells[:, 1])
    
            grid_x, grid_y = np.mgrid[min_x:max_x:self.n_cells_1d*(1j),
                    min_y:max_y:self.n_cells_1d*(1j)]
    
            gridded_data = scipy.interpolate.griddata(
                    self.cells, vals.reshape(-1), (grid_x, grid_y), method='cubic')
    
            plt.imshow(gridded_data.T, extent=(min_x,max_x,min_y,max_y),
                    origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.xticks([])
            plt.yticks([])
            if colorbar is not None:
                plt.colorbar()
        if self.n_dims == 1:
            # Generate regular gridding over which to plot.
            min_x = np.min(self.cells[:])
            max_x = np.max(self.cells[:])
    
            plt.plot(self.cells, vals)
            plt.xlim([min_x, max_x])

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=400)
            plt.close()
        else: plt.show()
