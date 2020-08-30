""" Sequentially compute IVR on small volcano.
Use updatable covariance.

"""
import os
import torch
import numpy as np
import volcapy.covariance.exponential as cl
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess
from volcapy.grid.grid_from_dsm import Grid
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud, _array_to_point_cloud
from volcapy.plotting.vtkutils import array_to_vector_cloud
import matplotlib.pyplot as plt

from timeit import default_timer as timer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

output_path = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results/small_update/"
data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_23823_cells"

# Indices of the data points that are along the coast.
from volcapy.data_preparation.paths import coast_data_inds

def main():
    os.makedirs(output_path, exist_ok=True)

    # Load
    F = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_full_surface.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()

    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"surface_data_coords.npy"))).float()
    deep_cells_inds = torch.from_numpy(
            np.load(os.path.join(data_folder, "deep_cells_inds.npy")))

    ground_truth = torch.from_numpy(
            np.load(os.path.join(data_folder, "post_sample.npy")))
    data_values = ground_truth

    # Dictionary between the original Niklas data and our discretization.
    niklas_data_inds = torch.from_numpy(
            np.load(os.path.join(data_folder, "niklas_data_inds_insurf.npy"))).long()

    # Plot Niklas data.
    niklas_coords = data_coords[niklas_data_inds].numpy()
    plt.scatter(data_coords[:, 0], data_coords[:, 1], c="k", alpha=0.1)
    plt.scatter(niklas_coords[:, 0], niklas_coords[:, 1], c=niklas_coords[:, 2])
    plt.scatter(niklas_coords[coast_data_inds, 0], niklas_coords[coast_data_inds, 1], c="r")
    plt.title("Paths on the Stromboli and location of coastal data.")
    plt.show()

    # Remove the coast data.
    coast_data_inds_infull = niklas_data_inds[coast_data_inds]
    non_coast_inds = np.delete(np.array(list(range(F.shape[0]))),
            coast_data_inds_infull)
    F_coast = F[coast_data_inds_infull, :]
    data_coast = data_values[coast_data_inds_infull]

    print("Size of inversion grid: {} cells.".format(volcano_coords.shape[0]))
    print("Number of datapoints: {}.".format(data_coords.shape[0]))
    size = data_coords.shape[0]*volcano_coords.shape[0]*8 / 1e9
    print("Size of Pushforward matrix: {} GB.".format(size))
    
    # Params
    data_std = 0.1
    lambda0 = 200.0
    sigma0 = 200.0
    m0 = 1300

    # Now ready to go to updatable covariance.
    from volcapy.update.updatable_covariance import UpdatableCovariance
    updatable_cov = UpdatableCovariance(cl, lambda0, sigma0, volcano_coords)

    from volcapy.update.updatable_covariance import UpdatableMean
    updatable_mean = UpdatableMean(m0 * torch.ones(volcano_coords.shape[0]),
            updatable_cov)

    # -----------------------
    # INVERT THE COASTAL DATA.
    # -----------------------
    start = timer()
    n_chunks = 5
    # Perform inversion in chunks.
    for i, (F_i, data_part) in enumerate(zip(
                torch.chunk(F_coast, chunks=n_chunks, dim=0),
                torch.chunk(data_coast, chunks=n_chunks, dim=0))):
        print("Processing data chunk nr {}.".format(i))
        updatable_cov.update(F_i, data_std)
        updatable_mean.update(data_part, F_i)
        m = updatable_mean.m.cpu().numpy()

    end = timer()
    print("Inversion of coastal data run in {} mins.".format((end - start)/60.0))

    """
    IVRs = []
    # Now compute the variance reduction associated to the other points.
    for i, F_i in enumerate(F_out):
        IVR = updatable_cov.compute_IVR(F_i.reshape(1, -1), data_std,
                integration_inds=deep_cells_inds)
        print(IVR)
        IVRs.append(IVR)

        # Save periodically.
        if i % 15 == 0:
            print("Saving at {}.".format(i))
            np.save(os.path.join(output_path, "IVRs.npy"), np.array(IVRs))

    np.save(os.path.join(output_path, "IVRs.npy"), np.array(IVRs))
    data_coords_out[:, 2] = data_coords_out[:, 2] + 50.0
    np.save(os.path.join(output_path, "data_coords_out.npy"), data_coords_out)
    # Add an offset for easier visualization.
    _array_to_point_cloud(data_coords_out.numpy(),
            np.array(IVRs),
            os.path.join(output_path, "IVRs.vtk"))
    """

if __name__ == "__main__":
    main()
