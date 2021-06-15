import torch
import numpy as np
from multiprocessing import Pool, RawArray

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')

from timeit import default_timer as timer

# This is a flag to tell the R interface which covariance model is being use.
# TODO: Implement a Kernel class to make things cleaner.
KERNEL_FAMILY = "matern32"


def compute_cov_pushforward(lambda0, F, cells_coords, device=None, n_chunks=200,
        n_flush=50):
    """ Compute the covariance pushforward.

    The covariance pushforward is just KF^T, where K is the model
    covariance matrix.

    Note that the sigam0^2 is not included, and one has to manually add it when
    using the covariance pushforward computed here.

    Parameters
    ----------
    lambda0: float
        Lenght-scale parameter
    F: tensor
        Forward operator matrix
    cells_coords: tensor
        n_cells * n_dims: cells coordinates
    device: toch.Device
        Device to perform the computation on, CPU or GPU.
    n_chunks: int
        Number of chunks to split the matrix into.
        Default is 200. Increase if get OOM errors.
    n_flush: int
        Synchronize threads and flush GPU cache every *n_flush* iterations.
        This is necessary to avoid OOM errors.
        Default is 50.

    Returns
    -------
    Tensor
        n_model * n_data covariance pushforward K F^t.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transfer everything to device.
    lambda0 = torch.tensor(lambda0, requires_grad=False).to(device)
    F = F.detach().to(device)
    cells_coords = cells_coords.detach().to(device)

    # Flush to make sure everything clean.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    inv_lambda2 = - np.sqrt(3) / lambda0
    n_dims = 3
    n_model = F.shape[1]

    # Array to hold the results. We will compute line by line and concatenate.
    tot = torch.Tensor().to(device)

    # Compute K * F^T chunk by chunk.
    # That is, of all the cell couples, we compute the distance between some
    # cells (here x) and ALL other cells. Then repeat for other chunk and
    # concatenate.
    for i, x in enumerate(torch.chunk(cells_coords, chunks=n_chunks, dim=0)):
        # Empty cache every so often. Otherwise we get out of memory errors.
        if i % n_flush == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Euclidean distance.
        d = torch.sqrt(torch.pow(
            x.unsqueeze(1).expand(x.shape[0], n_model, n_dims)
            - cells_coords.unsqueeze(0).expand(x.shape[0], n_model, n_dims)
            , 2).sum(2))
        tot = torch.cat((
                tot,
                torch.matmul(
                    torch.mul(
                            torch.ones(d.shape, device=device) - inv_lambda2 * d,
                            torch.exp(inv_lambda2 * d))
                    , F.t())))

    # Wait for all threads to complete.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    end = timer()

    return tot

def compute_diagonal(lambda0, cells_coords, device=None, n_chunks=200,
        n_flush=50):
    """ Compute the diagonal of the covariance matrix.

    Note that the sigam0^2 is not included, and one has to manually add it when
    using the covariance pushforward computed here.

    Parameters
    ----------
    lambda0: float
        Lenght-scale parameter
    cells_coords: tensor
        n_cells * n_dims: cells coordinates
    device: toch.Device
        Device to perform the computation on, CPU or GPU.
        Defaults to None, in which case first tries gpu and falls back to cpu
        if unavailable.
    n_chunks: int
        Number of chunks to split the matrix into.
        Default is 200. Increase if get OOM errors.
    n_flush: int
        Synchronize threads and flush GPU cache every *n_flush* iterations.
        This is necessary to avoid OOM errors.
        Default is 50.

    Returns
    -------
    Tensor (n_cells)
        Diagonal of the covariance matrix.
    """
    # We have a stationary covariance, so the diagonal is trivial.
    return torch.ones(cells_coords.shape[0])

def compute_cov(lambda0, cells_coords, i, j):
    """ Compute the covariance between two points.

    Note that, as always, sigma0 has been stripped.

    Parameters
    ----------
    lambda0: float
        Lenght-scale parameter
    cells_coords: tensor
        n_cells * n_dims: cells coordinates
    i: int
        Index of first cell (index in the cells_coords array).
    j: int
        Index of second cell.

    Returns
    -------
    Tensor
        (Stripped) covariance between cell nr i and cell nr j.
    """
    # Convert to torch.
    lambda0 = torch.tensor(lambda0, requires_grad=False)
    inv_lambda2 = - np.sqrt(3) / lambda0

    # Euclidean distance.
    d = torch.sqrt(torch.pow(
            cells_coords[i, :] - cells_coords[j, :], 2).sum())

    return (1 - inv_lambda2 * d) * torch.exp(inv_lambda2 * d)

def compute_full_cov(lambda0, cells_coords, device, n_chunks=200,
        n_flush=50):
    """ Compute the full covariance matrix.


    Note that the sigam0^2 is not included, and one has to manually add it when
    using the covariance pushforward computed here.

    Parameters
    ----------
    lambda0: float
        Lenght-scale parameter
    cells_coords: tensor
        n_cells * n_dims: cells coordinates
    device: toch.Device
        Device to perform the computation on, CPU or GPU.
    n_chunks: int
        Number of chunks to split the matrix into.
        Default is 200. Increase if get OOM errors.
    n_flush: int
        Synchronize threads and flush GPU cache every *n_flush* iterations.
        This is necessary to avoid OOM errors.
        Default is 50.

    Returns
    -------
    Tensor
        n_cells * n_cells covariance matrix.
    """
    # Transfer everything to device.
    lambda0 = torch.tensor(lambda0, requires_grad=False).to(device)
    cells_coords = cells_coords.to(device)

    inv_lambda2 = - np.sqrt(3) / lambda0
    n_dims = 3
    n_cells = cells_coords.shape[0]

    # Array to hold the results. We will compute line by line and concatenate.
    tot = torch.Tensor().to(device)

    # Compute K * F^T chunk by chunk.
    # That is, of all the cell couples, we compute the distance between some
    # cells (here x) and ALL other cells. Then repeat for other chunk and
    # concatenate.
    for i, x in enumerate(torch.chunk(cells_coords, chunks=n_chunks, dim=0)):
        # Empty cache every so often. Otherwise we get out of memory errors.
        if i % n_flush == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Euclidean distance.
        d = torch.sqrt(torch.pow(
            x.unsqueeze(1).expand(x.shape[0], n_cells, n_dims)
            - cells_coords.unsqueeze(0).expand(x.shape[0], n_cells, n_dims)
            , 2).sum(2))

        tot = torch.cat((
                tot,
                torch.mul(
                    torch.ones(d.shape, device=device) - inv_lambda2 * d,
                    torch.exp(inv_lambda2 * d))))

    # Wait for all threads to complete.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return tot

def compute_pushfwd_chunk(x, cells_coords, inv_lambda_2):
    """ Compute one chunk of the covariance pushforward.

    Returns
    -------
    A chunk of size x.shape[0], n_model of the covariance matrix.

    """
    # Put everything in Torch.
    x = torch.from_numpy(x).double()
    cells_coords = torch.from_numpy(cells_coords).double()
    n_model, n_dims = cells_coords.shape

    d = torch.sqrt(torch.pow(
        x.unsqueeze(1).expand(x.shape[0], n_model, n_dims)
        - cells_coords.unsqueeze(0).expand(x.shape[0], n_model, n_dims)
        , 2).sum(2))

    pushfwd_chunk = torch.mul(torch.ones(d.shape, dtype=torch.double) - inv_lambda_2 * d,
                    torch.exp(inv_lambda_2 * d))
    return pushfwd_chunk

def compute_cov_cpu(lambda0, coords, n_procs):
    """ Massively parallelized version (on CPU)

    """
    inv_lambda_2 = - np.sqrt(3) / lambda0

    n_cells = coords.shape[0]
    n_dim_coords = coords.shape[1]
    cov_shape = (n_cells, n_cells)
    cov_shared_buffer = RawArray('d', n_cells * n_cells)
    # Wrap as a numpy array so we can easily manipulates its data.
    cov_np = np.frombuffer(cov_shared_buffer).reshape(cov_shape)
    # Copy data to our shared array.
    np.copyto(cov_np, np.zeros(cov_shape))

    coords_shared_buffer = RawArray('d', n_cells * n_dim_coords)
    # Wrap as a numpy array so we can easily manipulates its data.
    coords_np = np.frombuffer(coords_shared_buffer).reshape(coords_shape)
    # Copy data to our shared array.
    np.copyto(coords_np, coords)

    # Start the process pool and do the computation.
    # Here we pass X and X_shape to the initializer of
    # each worker.
    # (Because X_shape is not a shared variable,
    # it will be copied to each
    # child process.)
    with Pool(processes=n_procs, initializer=init_worker,
            initargs=(cov_shared_buffer, cov_shape,
                    coords_shared_buffer, coords_shape,
                    inv_lambda_2)) as pool:
        result = pool.map(_worker_func, range(coords_shape[0]))

    return cov_np

# A global dictionary storing the variables passed from the initializer.
var_dict = {}
def init_worker(cov, cov_shape, coords, coords_shape, inv_lambda_2):
    var_dict['cov'] = cov
    var_dict['cov_shape'] = cov_shape
    var_dict['coords'] = coords
    var_dict['coords_shape'] = coords_shape
    var_dict['inv_lambda_2'] = inv_lambda_2

def _worker_func(i):
    cov_np = np.frombuffer(var_dict['cov']).reshape(var_dict['cov_shape'])
    coords_np = np.frombuffer(var_dict['coords']).reshape(var_dict['coords_shape'])

    # Still need to have two axes.
    cell = coords_np[i, :][None]

    # Compute one line of the pushfwd.
    tmp = compute_pushfwd_chunk(cell, coords_np, var_dict['inv_lambda_2'])
    cov_np[i, :] = tmp

    return None
