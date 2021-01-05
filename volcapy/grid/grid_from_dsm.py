""" Given a dsm, build a grid.

The grid will have the same xy resolution as the dsm.

"""
import numpy as np
import meshio
import pickle


class Grid():
    def __init__(self, cells, cells_roof, surface_inds, res_x, res_y, res_z):
        self.cells = cells
        self.cells_roof = cells_roof
        self.surface_inds = surface_inds
        self.res_x = res_x
        self.res_y = res_y
        self.res_z = res_z

    @classmethod
    def build_grid(cls, dsm_x, dsm_y, dsm_z, z_low, z_step):

        # Deduce resolutions
        res_x = np.abs(dsm_x[0] - dsm_x[1])
        res_y = np.abs(dsm_y[0] - dsm_y[1])

        res_z = z_step

        cells, cells_roof, surface_inds = build_grid_below_dsm(
                dsm_x, dsm_y, dsm_z, z_low, z_step)
        return cls(cells, cells_roof, surface_inds, res_x, res_y, res_z)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
                t = pickle.load(f)
                return cls(
                        t['cells'], t['cells_roof'], t['surface_inds'],
                        t['res_x'], t['res_y'], t['res_z'])

    def save(self, path):
        pickled_grid = {
                'cells': self.cells, 'cells_roof': self.cells_roof,
                'surface_inds': self.surface_inds,
                'res_x': self.res_x, 'res_y': self.res_y, 'res_z': self.res_z}

        with open(path, 'wb') as f:
                pickle.dump(pickled_grid, f)

    def __getitem__(self, index):
        return self.cells[index]

    @property
    def shape(self):
        return self.cells.shape

    @property
    def surface(self):
        return self.cells[self.surface_inds]

    def get_cell_hexahedron(self, cell_ind):
        """ Returns the coordinates of the corners of the hexahedron definint
        the current cells.

        """
        centroid_coord = self.__getitem__(cell_ind)
        corner_1 = centroid_coord + 1.0

    def generate_mesh_data(self, cell_data=None):
        """ Converts grid to a VTK UnstructuredGrid.

        """
        # Arrays of shape (self.n_cell, 3) containin the coords of the corners
        # of the corresponding hexahedron for each cells.
        bottom_front_left = self.cells + np.array([-self.res_x , -self.res_y, -self.res_z])
        bottom_front_right = self.cells + np.array([+self.res_x , -self.res_y, -self.res_z])

        bottom_back_left = self.cells + np.array([-self.res_x , +self.res_y, -self.res_z])
        bottom_back_right = self.cells + np.array([+self.res_x , +self.res_y, -self.res_z])

        top_front_left = self.cells + np.array([-self.res_x , -self.res_y, +self.res_z])
        top_front_right = self.cells + np.array([+self.res_x , -self.res_y, +self.res_z])

        top_back_left = self.cells + np.array([-self.res_x , +self.res_y, +self.res_z])
        top_back_right = self.cells + np.array([+self.res_x , +self.res_y, +self.res_z])

        points = np.hstack([bottom_front_left, bottom_front_right,
                bottom_back_left, bottom_back_right,
                top_front_left, top_front_right,
                top_back_left, top_back_right])
        # Black magic to put in correct order.
        points = points.reshape(1, -1).reshape(-1, 3)
        topology = np.array(list(range(points.shape[0])))
        topology = topology.reshape(-1, 8)

        """
        hex_mesh = meshio.Mesh(
                points,
                {"hexahedron": topology},
                cell_data=cell_data)

        # meshio.write("out.vtk", hex_mesh)
        """
        return points, topology


def build_grid_below_dsm(dsm_x, dsm_y, dsm_z, z_low, z_step):
    """ Given a dsm, discretize the space below it in cubic cells, down to some
    specified bottom level.

    """
    cells = []
    cells_roof = []
    surface_inds = []

    # Maximal altitude.
    z_max = np.max(dsm_z) + 2 * z_step
    z_levels = np.arange(z_low, z_max, z_step)
    for i, x in enumerate(dsm_x):
        for j, y in enumerate(dsm_y):
            # Edge case: if we are already above the dsm. Something is wrong.
            if dsm_z[i, j] < z_low:
                raise ValueError(""""Warning: the floor for z-modelling is 
                    above the dsm in some regions. This will create artificial
                    landmasses.
                    DSM altitude: {}.""".format(dsm_z[i, j]))
            for z in z_levels:
                cells.append([x, y, z])

                # Detect when we are at the surface and break.
                if np.abs(z - dsm_z[i, j]) <= z_step / 2:
                    # This time the cell is not a full square.
                    cells_roof.append(dsm_z[i, j])

                    # Get current index and save it.
                    current_ind = len(cells_roof) - 1
                    surface_inds.append(current_ind)
                    break
                else: cells_roof.append(z + z_step / 2.0)
    return (np.asarray(cells, dtype=np.float32),
            np.asarray(cells_roof, dtype=np.float32),
            np.asarray(surface_inds, dtype=np.int))
