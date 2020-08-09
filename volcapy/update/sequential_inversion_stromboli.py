""" Try the sequential inversion framework on synthetic data (conic volcano).

The synthetic data should be located in ../synthetic/synthetic_data/.

"""
import os
import torch
import numpy as np
import volcapy.covariance.exponential as cl
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess
from volcapy.grid.grid_from_dsm import Grid
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud, _array_to_vtk_point_cloud
from volcapy.plotting.vtkutils import array_to_vtk_vector_cloud

from timeit import default_timer as timer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load
    data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/input_data"
    F = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_niklas.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(
            grid.cells).float().detach()

    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"niklas_data_coords.npy"))).float()
    data_values = torch.from_numpy(
            np.load(os.path.join(data_folder,"niklas_data_obs.npy"))).float()


    # Subdivide for variance reduction computation.
    part_ind = 60
    F_rest = F[part_ind:, :]
    F = F[:part_ind, :]

    data_coords_part = data_coords[part_ind:, :]
    data_coords = data_coords[:part_ind, :]

    data_values = data_values[:part_ind]

    print("Size of inversion grid: {} cells.".format(volcano_coords.shape[0]))
    print("Number of datapoints: {}.".format(data_coords.shape[0]))
    size = data_coords.shape[0]*volcano_coords.shape[0]*8 / 1e9
    print("Size of Pushforward matrix: {} GB.".format(size))
    
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

    n_chunks = 5
    # Loop over measurement chunks.
    for i, (F_i, data_part) in enumerate(zip(
                torch.chunk(F, chunks=n_chunks, dim=0),
                torch.chunk(data_values, chunks=n_chunks, dim=0))):
        print("Processing data chunk nr {}.".format(i))
        updatable_cov.update(F_i, data_std)
        updatable_mean.update(data_part, F_i)
        m = updatable_mean.m.cpu().numpy()
        np.save("m_post_{}_stromboli.npy".format(i), m)

        irregular_array_to_point_cloud(volcano_coords.numpy(), m,
                "m_post_{}_stromboli.vtk".format(i), fill_nan_val=-20000.0)

    end = timer()
    print("Sequential inversion run in {} s.".format(end - start))

    # Save the variance.
    variances = updatable_cov.extract_variance()
    irregular_array_to_point_cloud(volcano_coords.numpy(),
            variances.detach().cpu().numpy(),
            "variances_stromboli.vtk", fill_nan_val=-20000.0)

    # Save the coordinates of the observation points for later plotting.
    # The point data will be used for glyph orientation when plotting.
    orientation_data = np.zeros((data_coords.shape[0], 3))
    orientation_data[:, 2] = -1.0

    # Add an offset for easier visualization.
    data_coords[:, 2] = data_coords[:, 2] + 50.0
    array_to_vtk_vector_cloud(data_coords.numpy(),
            orientation_data,
            "data_points.vtk")

    IVRs = []
    # Now compute the variance reduction associated to the other points.
    for i, F_i in enumerate(F_rest):
        IVR = updatable_cov.compute_IVR(F_i.reshape(1, -1), data_std)
        print(IVR)
        IVRs.append(IVR)

        # Save periodically.
        if i % 15 == 0:
            print("Saving at {}.".format(i))
            np.save("IVRs.npy", np.array(IVRs))

    np.save("IVRs.npy", np.array(IVRs))
    data_coords_part[:, 2] = data_coords_part[:, 2] + 50.0
    np.save("data_coords_IVR.npy", data_coords_part)
    # Add an offset for easier visualization.
    _array_to_vtk_point_cloud(data_coords_part.numpy(),
            np.array(IVRs),
            "IVRs.vtk")

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
