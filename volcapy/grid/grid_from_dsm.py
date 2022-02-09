""" Given a dsm, build a grid.

The grid will have the same xy resolution as the dsm.

"""
import numpy as np
import pickle


class Grid():
    def __init__(self, cells, cells_roof, surface_inds, mesh_index_table=None):
        self.cells = cells
        self.cells_roof = cells_roof
        self.surface_inds = surface_inds

        # Deduce resolutions.
        sorted_unique_x = np.sort(list(set(cells[:, 0])))
        sorted_unique_y = np.sort(list(set(cells[:, 1])))
        sorted_unique_z = np.sort(list(set(cells[:, 2])))
        self.res_x = sorted_unique_x[1] - sorted_unique_x[0]
        self.res_y = sorted_unique_y[1] - sorted_unique_y[0]
        self.res_z = sorted_unique_z[1] - sorted_unique_z[0]

        if mesh_index_table is None: mesh_index_table = self.gen_mesh_index_table()
        self.mesh_index_table = mesh_index_table

    @classmethod
    def build_grid(cls, dsm_x, dsm_y, dsm_z, z_low, z_step):

        # Deduce resolutions
        res_x = np.abs(dsm_x[0] - dsm_x[1])
        res_y = np.abs(dsm_y[0] - dsm_y[1])

        res_z = z_step

        cells, cells_roof, surface_inds = build_grid_below_dsm(
                dsm_x, dsm_y, dsm_z, z_low, z_step)
        return cls(cells, cells_roof, surface_inds)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
                t = pickle.load(f)
                return cls(
                        t['cells'], t['cells_roof'], t['surface_inds'],
                        mesh_index_table=t['mesh_index_table'])

    def save(self, path):
        pickled_grid = {
                'cells': self.cells, 'cells_roof': self.cells_roof,
                'surface_inds': self.surface_inds,
                'mesh_index_table': self.mesh_index_table
                }

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

    def build_mesh(self):
        """ Buils a regular mesh containing the grid points. 
        This is useful for plotting and interpolation.

        The grids we are working with are irregular due to the shape of the domain, 
        i.e. compared to a full parallelepipedic mesh, some cells are missing. 
        The goal of this function is to build a parallelepipedic mesh that encloses 
        the grid. 

        Returns
        -------
        X_mesh, Y_,mesh, Z_mesh: array [n_x, n_y, n_z]
            Mesh arrays. The number of points along each dimensions corresponds 
            to the number of unique values along that dimension in the original grid array.

        """
        x_s = np.sort(list(set(self.cells[:, 0])))
        y_s = np.sort(list(set(self.cells[:, 1])))
        z_s = np.sort(list(set(self.cells[:, 2])))

        # 3D mesh grid for interpolation
        X_mesh, Y_mesh, Z_mesh = np.meshgrid(x_s, y_s, z_s, indexing='ij')
        return X_mesh, Y_mesh, Z_mesh

    def gen_mesh_index_table(self):
        """ Generates a table that gives the corresponding indices in the regular 
        mesh for each cell in the grid. The regular mesh is the one build by 
        self.build_mesh.

        Returns
        -------
        mesh_index_table: array (n_cells, n_dims)

        """
        X_mesh, Y_mesh, Z_mesh = self.build_mesh()

        # Idea of the algo: the cells array is a bit like the mesh 
        # array, but with cells removed. Hence we can start by iterating the two 
        # synchronously and increment the mesh index faster.
        mesh_size = X_mesh.shape[0]*X_mesh.shape[1]*X_mesh.shape[2]
        start_ind_mesh = 0
        regular_index_table = []
        for ind_1d, cell in enumerate(self.cells):
            print(ind_1d)
            if start_ind_mesh >= mesh_size:
                return
            for ind_mesh in range(start_ind_mesh, mesh_size):
                # Indices in meshed array.
                i, j, k = np.unravel_index(ind_mesh, X_mesh.shape)
                pt_mesh = np.array([X_mesh[i, j, k], Y_mesh[i, j, k], Z_mesh[i, j, k]])
                print(ind_mesh)
                print(cell)
                print(pt_mesh)

                # If close enough break the search.
                start_ind_mesh += 1
                dist = np.linalg.norm(pt_mesh - cell)
                print(dist)
                if (dist < np.min([self.res_x, self.res_y, self.res_z]) / 3):
                    regular_index_table.append((i, j, k))
                    break

        return np.array(regular_index_table)

    def mesh_values(self, values):
        """ Given a list of values at grid points, returns them in a regular mesh, 
        as built by self.build_mesh.

        Returns
        -------
        X_mesh: array [n_x, n_y, n_z]
        Y_mesh: array [n_x, n_y, n_z]
        Z_mesh: array [n_x, n_y, n_z]
        vals_mesh: array [n_x, n_y, n_z]

        """
        X_mesh, Y_mesh, Z_mesh = self.build_mesh()
        vals_mesh = np.full(X_mesh.shape, np.nan)
        vals_mesh[
                self.mesh_index_table[:, 0],
                self.mesh_index_table[:, 1],
                self.mesh_index_table[:, 2]] = values.reshape(-1)
        return vals_mesh

    # TODO: Unused. Check and remove.
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
