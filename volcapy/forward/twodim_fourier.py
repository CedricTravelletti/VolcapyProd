""" Discrete Fourier transform operator in 2D.

"""
import numpy as np
from multiprocessing import Pool, RawArray
from volcapy.niklas.banerjee2d import banerjee


# =====================
# Multiprocessing Stuff
# =====================
# A global dictionary storing the variables passed from the initializer.
var_dict = {}
def init_worker(F_re, F_im, F_shape, coords, coords_shape, meta):
    var_dict['F_re'] = F_re
    var_dict['F_im'] = F_im
    var_dict['F_shape'] = F_shape
    var_dict['coords'] = coords
    var_dict['coords_shape'] = coords_shape
    var_dict['meta'] = meta

def _worker_func(i):
    F_np_re = np.frombuffer(var_dict['F_re'], dtype=np.float32).reshape(var_dict['F_shape'])
    F_np_im = np.frombuffer(var_dict['F_im'], dtype=np.float32).reshape(var_dict['F_shape'])
    coords_np = np.frombuffer(var_dict['coords']).reshape(var_dict['coords_shape'])

    M = var_dict['meta']['M']
    N = var_dict['meta']['N']

    cell = coords_np[i, :]
    tmp_re, tmp_im = _compute_forward_column(cell, M, N)
    F_np_re[:, i] = tmp_re
    F_np_im[:, i] = tmp_im
    return None

def compute_forward(coords, M, N, n_procs):
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

    Returns
    -------
    ndarray
        Forward operator, size n_data * n_cells.

    """
    n_cells = coords.shape[0]
    n_dim_coords = coords.shape[1]
    F_shape = (n_data, n_cells)
    coords_shape = (n_cells, n_dim_coords)

    meta = {'M': M, 'N': N, }

    # ----------------------------
    # Prepare the parallelization.
    # ----------------------------
    F_shared_buffer_re = RawArray('f', M*N * n_cells)
    F_shared_buffer_im = RawArray('f', M*N * n_cells)
    # Wrap as a numpy array so we can easily manipulates its data.
    F_np_re = np.frombuffer(F_shared_buffer_re, dtype=np.float32).reshape(F_shape)
    F_np_im = np.frombuffer(F_shared_buffer_im, dtype=np.float32).reshape(F_shape)
    # Copy data to our shared array.
    np.copyto(F_np_re, np.zeros(F_shape))
    np.copyto(F_np_im, np.zeros(F_shape))

    coords_shared_buffer = RawArray('d', n_cells * n_dim_coords)
    # Wrap as a numpy array so we can easily manipulates its data.
    coords_np = np.frombuffer(coords_shared_buffer).reshape(coords_shape)
    # Copy data to our shared array.
    np.copyto(coords_np, coords)

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
            initargs=(F_shared_buffer_re, F_shared_buffer_im, F_shape,
                    coords_shared_buffer, coords_shape,
                    meta)) as pool:
        result = pool.map(_worker_func, range(coords_shape[0]))

    return F_np_re, F_np_im

def _compute_forward_column(cell, M, N):
    """ Helper function for parallelizing the computation of the forward.

    """
    F_column_re = np.zeros(M*N, dtype=np.float32)
    F_column_im = np.zeros(M*N, dtype=np.float32)

    for i in range(M):
        for j in range(N):
            # Index in a 1D array of frequencies.
            ind = np.ravel_multi_index((i, j), (M, N))
            F_column_re[ind] = float(
                    np.cos(np.pi / M * i * cell[0]))
            F_column_im[ind] = float(
                    np.sin(np.pi / N * j * cell[1]))
    return F_column_re, F_column_im
