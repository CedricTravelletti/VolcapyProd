import torch
import numpy as np

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')

from timeit import default_timer as timer

# This is a flag to tell the R interface which covariance model is being use.
# TODO: Implement a Kernel class to make things cleaner.
KERNEL_FAMILY = "squared exponential"

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
    Tensor
        n_model * n_data covariance pushforward K F^t.
    """
    start = timer()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transfer everything to device.
    lambda0 = torch.tensor(lambda0, requires_grad=False).to(device)
    F = F.to(device)
    cells_coords = cells_coords.to(device)

    inv_lambda2 = - 1 / (2 * lambda0**2)
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

        # Squared euclidean distance.
        d_2 = torch.pow(
            x.unsqueeze(1).expand(x.shape[0], n_model, n_dims)
            - cells_coords.unsqueeze(0).expand(x.shape[0], n_model, n_dims)
            , 2).sum(2)
        tot = torch.cat((
                tot,
                torch.matmul(
                    torch.exp(inv_lambda2 * d_2)
                    , F.t())))

    # Wait for all threads to complete.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    end = timer()
    print((end - start)/60.0)

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
    inv_lambda2 = - 1 / (2 * lambda0**2)

    # Squared euclidean distance.
    d_2 = torch.pow(
            cells_coords[i, :] - cells_coords[j, :]
            , 2).sum()

    return torch.exp(inv_lambda2 * d_2)

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
    lambda0 = lambda0.to(device)
    cells_coords = cells_coords.to(device)

    inv_lambda2 = - 1 / (2 * lambda0**2)

    # WARNING: ONLY ADAPTED TO 2 DIMS.
    n_dims = 2
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

        # Squared euclidean distance.
        d_2 = torch.pow(
            x.unsqueeze(1).expand(x.shape[0], n_cells, n_dims)
            - cells_coords.unsqueeze(0).expand(x.shape[0], n_cells, n_dims)
            , 2).sum(2)
        tot = torch.cat((
                tot,
                torch.exp(inv_lambda2 * d_2)
                ))

    # Wait for all threads to complete.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


    return tot

def compute_full_cov_gradient(lambda0, cells_coords, device, n_chunks=200,
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
    lambda0 = lambda0.to(device)
    cells_coords = cells_coords.to(device)

    inv_lambda2 = - 1 / (2 * lambda0**2)

    # WARNING: ONLY ADAPTED TO 2 DIMS.
    n_dims = 2
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

        # Squared euclidean distance.
        d_2 = torch.pow(
            x.unsqueeze(1).expand(x.shape[0], n_cells, n_dims)
            - cells_coords.unsqueeze(0).expand(x.shape[0], n_cells, n_dims)
            , 2).sum(2)
        tot = torch.cat((
                tot,
                torch.exp(inv_lambda2 * d_2)
                ))

    # SEPCIAL FOR DERIVATIVE.
    tot = 2 * inv_lambda2 * tot
    
    # Wait for all threads to complete.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


    return tot


def compute_full_sqeuclidean_cov(cells_coords, device, n_chunks=200,
        n_flush=50):
    """ Compute the full matrix of pairwise squared euclidean distances.

    Parameters
    ----------
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
    cells_coords = cells_coords.to(device)

    # WARNING: ONLY ADAPTED TO 2 DIMS.
    n_dims = 2
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

        # Squared euclidean distance.
        d_2 = torch.pow(
            x.unsqueeze(1).expand(x.shape[0], n_cells, n_dims)
            - cells_coords.unsqueeze(0).expand(x.shape[0], n_cells, n_dims)
            , 2).sum(2)
        tot = torch.cat((
                tot,
                d_2
                ))

    # Wait for all threads to complete.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return tot
