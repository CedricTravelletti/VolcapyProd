""" Try the sequential inversion framework on synthetic data (conic volcano).

The synthetic data should be located in ../synthetic/synthetic_data/.

"""
import os
import torch
import numpy as np
import volcapy.covariance.exponential as cl
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud

from timeit import default_timer as timer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load
    data_folder = "../synthetic/synthetic_data/"
    F = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_synth.npy"))).float()
    volcano_coords = torch.from_numpy(
            np.load(os.path.join(data_folder, "cells_coords.npy"))).float()
    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"data_coords_synth.npy"))).float()
    data_values = torch.from_numpy(
            np.load(os.path.join(data_folder,"data_values_synth.npy"))).float()
    ground_truth = torch.from_numpy(
            np.load(os.path.join(data_folder,"density_synth.npy"))).float()

    print("Size of inversion grid: {} cells.".format(volcano_coords.shape[0]))
    print("Number of datapoints: {}.".format(data_coords.shape[0]))
    size = data_coords.shape[0]*volcano_coords.shape[0]*8 / 1e9
    print("Size of Pushforward matrix: {} GB.".format(size))

    # Synthetic volcano grid parameters.
    nx = 120
    ny = 120
    nz = 60
    res_x = 50
    res_y = 50
    res_z = 50

    # Partition the data.
    # Test-Train split.
    n_keep = 300
    F_part_1 = F[:n_keep, :]
    F_part_2 = F[n_keep:600, :]

    data_1 = data_values[:n_keep]
    data_2 = data_values[n_keep:600]
    
    # Params
    data_std = 0.1
    lambda0 = 200.0
    sigma0 = 200.0
    m0 = 1300

    start = timer()

    # Now ready to go to updatable covariance.
    from volcapy.update.updatable_covariance import UpdatableCovariance
    updatable_cov = UpdatableCovariance(cl, lambda0, sigma0, volcano_coords)

    from volcapy.update.updatable_covariance import UpdatableMean
    updatable_mean = UpdatableMean(m0 * torch.ones(volcano_coords.shape[0]),
            updatable_cov)

    # Loop over measurement chunks.
    for i, (F_part, data_part) in enumerate(zip(
                torch.chunk(F, chunks=20, dim=0),
                torch.chunk(data_values, chunks=20, dim=0))):
        print("Processing data chunk nr {}.".format(i))
        updatable_cov.update(F_part, data_std)
        updatable_mean.update(data_part, F_part)
        m = updatable_mean.m.cpu().numpy()
        np.save("m_post_{}.npy".format(i), m)

        irregular_array_to_point_cloud(volcano_coords.numpy(), m,
                "m_post_{}.vtk".format(i), fill_nan_val=-20000.0)

    end = timer()
    print("Sequential inversion run in {} s.".format(end - start))

    # Save the variance.
    variances = updatable_cov.extract_variance()
    irregular_array_to_point_cloud(volcano_coords.numpy(),
            variances.detach().cpu().numpy(),
            "variances.vtk", fill_nan_val=-20000.0)

    """
    # Run inversion in one go to compare.
    import volcapy.covariance.exponential as kernel
    start = timer()
    myGP = InverseGaussianProcess(m0, sigma0, lambda0,
            volcano_coords, kernel,
            logger=logger)

    cov_pushfwd = cl.compute_cov_pushforward(
            lambda0, F, volcano_coords, n_chunks=20,
            n_flush=50)
    
    m_post_m, m_post_d = myGP.condition_model(F, data_values, data_std,
            concentrate=False,
            is_precomp_pushfwd=False)

    end = timer()
    print("Non-sequential inversion run in {} s.".format(end - start))

    # Compute RMSE for each.
    train_rmse_seq = torch.sqrt(torch.mean((data_values - F @ m)**2))
    train_rmse_nonseq = torch.sqrt(torch.mean((data_values - F @ m_post_m)**2))

    print("Sequential train RMSE {}".format(train_rmse_seq.item()))
    print("Non-sequential train RMSE {}".format(train_rmse_nonseq.item()))
    
    np.save("m_post_nonsequential.npy", m_post_m.detach().cpu().numpy())
    
    irregular_array_to_point_cloud(volcano_coords.numpy(),
            m_post_m.detach().cpu().numpy(),
            "m_post_nonsequential.vtk", fill_nan_val=-20000.0)
    """

if __name__ == "__main__":
    main()
