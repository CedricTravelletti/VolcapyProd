""" Compute gravimetric forward operator for a given geometry.

"""
# TODO: All the parallelization stuff should be centralized in a single class
# for all the different type of forwards.
# The only difficulty that preventss us from doing it now is that different
# types of forward may take different arguments and thus have different number
# of arguments and variable names for the worker fun.
# We should try to find a solution in which we only redefine the compute column
# function for the different forwards and the argument handling is automatized.
import numpy as np
from multiprocessing import Pool, RawArray
from volcapy.niklas.banerjee import banerjee


# Gravitational constant.
G = 6.67e-6       #Transformation factor to get result in mGal

# =====================
# Multiprocessing Stuff
# =====================
# A global dictionary storing the variables passed from the initializer.
var_dict = {}
def init_worker(F, F_shape, coords, coords_shape,
        roofs, roofs_shape, data, data_shape, meta):
    var_dict['F'] = F
    var_dict['F_shape'] = F_shape
    var_dict['coords'] = coords
    var_dict['coords_shape'] = coords_shape
    var_dict['roofs'] = roofs
    var_dict['roofs_shape'] = roofs_shape
    var_dict['data_coords'] = data
    var_dict['data_coords_shape'] = data_shape
    var_dict['meta'] = meta

def _worker_func(i):
    F_np = np.frombuffer(var_dict['F'], dtype=np.float32).reshape(var_dict['F_shape'])
    coords_np = np.frombuffer(var_dict['coords']).reshape(var_dict['coords_shape'])
    roofs_np = np.frombuffer(var_dict['roofs']).reshape(var_dict['roofs_shape'])
    data_np = np.frombuffer(var_dict['data_coords']).reshape(var_dict['data_coords_shape'])

    res_x = var_dict['meta']['res_x']
    res_y = var_dict['meta']['res_y']
    res_z = var_dict['meta']['res_z']

    cell = coords_np[i, :]
    cell_roof = roofs_np[i]
    tmp = _compute_forward_column(
            cell, cell_roof, res_x, res_y, res_z, data_np)
    F_np[:, i] = tmp
    return None

def compute_forward(coords, roofs, res_x, res_y, res_z, data_coords, n_procs):
    """ Compute the forward operator associated to a given topography/irregular
    grid. In the end, it only need a list of cells.

    Parameters
    ----------
    coords: (n_cell, n_dims) ndarray
        Cells centroid coordinates.
    roofs: (n_cell) ndarray
        For each cell, the absolute altitude of its roof.
        This is mainly useful for surface cells, which intersect the
        topography and hence are not full prisms.
    res_x: float
        Length of a cell in x-direction (meters).
    res_y_float
        Length of a cell in y_direction (meters).
    res_z: float
        Length of a cell in z_direction (meters).
    data_coords: (n_data, n_dims) ndarray
        Data measurements coordinates.
    n_procs: int
        Number of processes to use to parallelize computation.

    Returns
    -------
    ndarray
        Forward operator, size n_data * n_cells.

    """
    n_cells = coords.shape[0]
    n_dim_coords = coords.shape[1]
    n_data = data_coords.shape[0]
    n_dim_data = data_coords.shape[1]
    F_shape = (n_data, n_cells)
    data_shape = (n_data, n_dim_data)
    coords_shape = (n_cells, n_dim_coords)
    roofs_shape = (n_cells)

    meta = {'res_x': res_x, 'res_y': res_y, 'res_z': res_z, }

    # ----------------------------
    # Prepare the parallelization.
    # ----------------------------
    F_shared_buffer = RawArray('f', n_data * n_cells)
    # Wrap as a numpy array so we can easily manipulates its data.
    F_np = np.frombuffer(F_shared_buffer, dtype=np.float32).reshape(F_shape)
    # Copy data to our shared array.
    np.copyto(F_np, np.zeros(F_shape))

    coords_shared_buffer = RawArray('d', n_cells * n_dim_coords)
    # Wrap as a numpy array so we can easily manipulates its data.
    coords_np = np.frombuffer(coords_shared_buffer).reshape(coords_shape)
    # Copy data to our shared array.
    np.copyto(coords_np, coords)

    # Same with roofs
    roofs_shared_buffer = RawArray('d', n_cells)
    # Wrap as a numpy array so we can easily manipulates its data.
    roofs_np = np.frombuffer(roofs_shared_buffer).reshape(roofs_shape)
    # Copy data to our shared array.
    np.copyto(roofs_np, roofs)

    # Same with data, so noesnt need to be copied along processes.
    data_coords_shared_buffer = RawArray('d', n_data * n_dim_data)
    # Wrap as a numpy array so we can easily manipulates its data.
    data_coords_np = np.frombuffer(data_coords_shared_buffer).reshape(data_shape)
    # Copy data to our shared array.
    np.copyto(data_coords_np, data_coords)


    # Start the process pool and do the computation.
    # Here we pass X and X_shape to the initializer of
    # each worker.
    # (Because X_shape is not a shared variable,
    # it will be copied to each
    # child process.)
    with Pool(processes=n_procs, initializer=init_worker,
            initargs=(F_shared_buffer, F_shape,
                    coords_shared_buffer, coords_shape,
                    roofs_shared_buffer, roofs_shape,
                    data_coords_shared_buffer, data_shape, meta)) as pool:
        result = pool.map(_worker_func, range(coords_shape[0]))

    return F_np

def _compute_forward_column(cell, cell_roof, res_x, res_y, res_z, data_coords):
    """ Helper function for parallelizing the computation of the forward.

    """
    n_data = data_coords.shape[0]
    F_column = np.zeros(n_data, dtype=np.float32)

    for j, data in enumerate(data_coords):
        # Compute cell endpoints.
        xh = cell[0] + res_x / 2.0
        xl = cell[0] - res_x / 2.0
        yh = cell[1] + res_y / 2.0
        yl = cell[1] - res_y / 2.0
        zh = cell_roof
        zl = cell[2] - res_z / 2.0

        F_column[j] = float(G * banerjee(
                xh, xl, yh, yl, zh, zl,
                data[0],  data[1],  data[2]))
    return F_column
