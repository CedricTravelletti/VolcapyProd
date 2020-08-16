""" Compute IVRs on Niklas Stromboli data.
We use a small batch of data for conditioning and then compute IVr for each of
the remaining data locations.

This script compute the regionalized IVR not for points, but for different sets
of paths.

"""
import os
import torch
import numpy as np
import volcapy.covariance.exponential as cl
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess
from volcapy.grid.grid_from_dsm import Grid
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud, _array_to_point_cloud
from volcapy.plotting.vtkutils import array_to_vector_cloud

from timeit import default_timer as timer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

output_path = "/home/cedric/PHD/Dev/VolcapySIAM/volcapy/update/regionalized_IVR_results_paths"

# Indices of the data points that are along the coast.
from volcapy.data_preparation.paths import coast_data_inds

def main():
    # Load
    deep_cells_inds = torch.from_numpy(
            np.load(
            "/home/cedric/PHD/Dev/VolcapySIAM/data/inversion_data_dsm_coarse/deep_cells_inds.npy")
            ).long()
    data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/inversion_data_dsm_coarse"
    F_full = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_niklas.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(
            grid.cells).float().detach()

    data_coords_full = torch.from_numpy(
            np.load(os.path.join(data_folder,"niklas_data_coords.npy"))).float()
    data_values_full = torch.from_numpy(
            np.load(os.path.join(data_folder,"niklas_data_obs.npy"))).float()


    # Subdivide for variance reduction computation.
    F = F_full[coast_data_inds, :]
    data_coords = data_coords_full[coast_data_inds, :]
    data_values = data_values_full[coast_data_inds]

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

    end = timer()
    print("Sequential inversion run in {} s.".format(end - start))

    # Save the coordinates of the observation points for later plotting.
    # The point data will be used for glyph orientation when plotting.
    orientation_data = np.zeros((data_coords.shape[0], 3))
    orientation_data[:, 2] = -1.0

    # Add an offset for easier visualization.
    data_coords[:, 2] = data_coords[:, 2] + 50.0
    array_to_vector_cloud(data_coords.numpy(),
            orientation_data,
            os.path.join(output_path, "data_points.vtk"))

    # Loop over paths and compute variance reduction associated to each.
    from volcapy.data_preparation.paths import paths
    for i, path_inds in enumerate(paths):
        print("Processing path {} / {}.".format(i, len(paths)))
        IVR = updatable_cov.compute_IVR(F_full[path_inds, :], data_std,
                integration_inds=deep_cells_inds)
        print("IVR for path {}: {}".format(i, IVR))
        _array_to_point_cloud(data_coords_full[path_inds].numpy(),
            np.full(len(path_inds), IVR, dtype=np.float32),
            os.path.join(output_path, "IVR_path{}.vtk".format(i)))

if __name__ == "__main__":
    main()
