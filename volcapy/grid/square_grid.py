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


class Grid():
    def __init__(self, n_dims, n_cells_1d):
        self.res = 2.0 / n_cells_1d
        n_cells = n_cells_1d**n_dims

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
        self.cells = cells

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
