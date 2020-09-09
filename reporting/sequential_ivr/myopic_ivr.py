""" Run myopic excursion set reconstruction strategy using weighted IVR.

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

output_path = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results/myopic_23823"
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
    niklas_coords = data_coords[niklas_data_inds].numpy()

    # PATHS ON THE VOLCANO.
    from volcapy.data_preparation.paths import paths as paths_niklas
    # Convert to indices in the full dataset.
    paths = []
    for path in paths_niklas:
        paths.append(niklas_data_inds[path].long())

    # --------------------------------
    # DEFINITION OF THE EXCURSION SET.
    # --------------------------------
    THRESHOLD = 700.0
    excursion_inds = (ground_truth >= THRESHOLD).nonzero()[:, 0]

    # Plot situation.
    plt.scatter(data_coords[:, 0], data_coords[:, 1], c="k", alpha=0.1)

    plt.scatter(
            volcano_coords[excursion_inds, 0],
            volcano_coords[excursion_inds, 1], c="r", alpha=0.07)

    plt.scatter(niklas_coords[:, 0], niklas_coords[:, 1],
            c=niklas_coords[:, 2])
    plt.scatter(niklas_coords[coast_data_inds, 0], niklas_coords[coast_data_inds, 1], c="r")

    for i, path in enumerate(paths):
        for x, y in zip(data_coords[path, 0], data_coords[path, 1]):
                plt.text(x, y, str(i), color="black", fontsize=6)

    plt.title("Paths on the Stromboli, location of coastal data and excursion set.")
    plt.show()

    # Remove the coast data.
    coast_data_inds_infull = niklas_data_inds[coast_data_inds]
    non_coast_inds = np.delete(np.array(list(range(F.shape[0]))),
            coast_data_inds_infull)
    F_coast = F[coast_data_inds_infull, :]
    data_coast = data_values[coast_data_inds_infull]

    # Get Niklas data
    F_niklas = F[niklas_data_inds, :]
    data_niklas = data_values[niklas_data_inds]
    F_niklas = F_niklas[np.delete(np.array(list(range(F_niklas.shape[0]))),
            coast_data_inds), :]
    data_niklas = data_niklas[np.delete(np.array(list(range(data_niklas.shape[0]))),
            coast_data_inds)]

    print("Size of inversion grid: {} cells.".format(volcano_coords.shape[0]))
    print("Number of datapoints: {}.".format(data_coords.shape[0]))
    size = data_coords.shape[0]*volcano_coords.shape[0]*8 / 1e9
    print("Size of Pushforward matrix: {} GB.".format(size))
    
    # Params
    data_std = 0.1
    # lambda0 = 338.0
    lambda0 = 270.0
    sigma0 = 314.0
    m0 = -150.0

    # Define GP model.
    updatable_gp = UpdatableGP(cl, lambda0, sigma0, m0, volcano_coords,
            n_chunks=10)

    # -----------------------
    # INVERT THE COASTAL DATA.
    # -----------------------
    start = timer()
    n_chunks = 5
    # Perform inversion in chunks.
    for i, (F_i, data_part) in enumerate(zip(
                torch.chunk(F_coast, chunks=n_chunks, dim=0),
                torch.chunk(data_coast, chunks=n_chunks, dim=0))):
        print("Processing costal data chunk nr {}.".format(i))
        updatable_gp.update(F_i, data_part, data_std)

    end = timer()
    print("Inversion of coastal data run in {} mins.".format((end - start)/60.0))

    # PLOT CURRENT ESTIMATION OF THE EXCURSION SET.
    excursion_inds_est = (updatable_gp.mean_vec >= 500.0).nonzero()[:, 0]

    plt.scatter(
            volcano_coords[excursion_inds, 0],
            volcano_coords[excursion_inds, 1], c="r", alpha=0.07,
            s=60)
    plt.scatter(
            volcano_coords[excursion_inds_est, 0],
            volcano_coords[excursion_inds_est, 1], c="g", alpha=0.07)

    plt.savefig("plugin_est.png", bbox_inches='tight', pad_inches=0, dpi=400)
    plt.close()
    # plt.show()

    # --------------------------
    # COMPUTE IVR FOR EACH PATH.
    # --------------------------
    # First have to extract the variance information to compute the weights.
    from volcapy.gaussian_cdf import gaussian_cdf
    current_variance = updatable_gp.covariance.extract_variance(n_chunks=5)
    lower = torch.tensor([THRESHOLD])
    weights = gaussian_cdf(updatable_gp.mean_vec, current_variance.reshape(-1, 1), lower=lower)

    plt.scatter(volcano_coords[:, 0], volcano_coords[:, 1], c=weights,
            alpha=0.2)
    plt.scatter(
            volcano_coords[excursion_inds, 0],
            volcano_coords[excursion_inds, 1], c="r", alpha=0.07,
            s=10)

    plt.savefig("weights.png", bbox_inches='tight', pad_inches=0, dpi=400)
    plt.close()
    # plt.show()

    IVRs = []
    # Now compute the variance reduction associated to each path.
    for i, path in enumerate(paths):
        IVR = updatable_gp.IVR(F[path, :], data_std,
                # integration_inds=deep_cells_inds,
                weights=weights)
        print(IVR)
        IVRs.append(IVR)

    # Plot the results.
    ivrs_min = np.min(IVRs)
    ivrs_max = np.max(IVRs)
    plt.scatter(
            volcano_coords[excursion_inds, 0],
            volcano_coords[excursion_inds, 1], c="r", alpha=0.07,
            s=60)
    for i, path in enumerate(paths):
        plt.scatter(data_coords[path, 0], data_coords[path, 1],
                c=IVRs[i]*np.ones(len(path)), cmap="inferno",
                vmin=ivrs_min, vmax=ivrs_max)
    plt.colorbar()

    plt.savefig("ivrs_paths.png", bbox_inches='tight', pad_inches=0, dpi=400)
    plt.close()
    # plt.show()
    np.save("weighted_IVRs_paths.npy", IVRs)

    # ---------------------------
    # DO THE SAME, BUT POINTWISE.
    # ---------------------------
    IVRs = []
    # Now compute the variance reduction associated to each path.
    for i, ind in enumerate(niklas_data_inds):
        IVR = updatable_gp.IVR(F[ind, :].reshape(1, -1), data_std,
                # integration_inds=deep_cells_inds,
                weights=weights)
        print(IVR)
        IVRs.append(IVR)

    # Plot the results.
    ivrs_min = np.min(IVRs)
    ivrs_max = np.max(IVRs)
    plt.scatter(
            volcano_coords[excursion_inds, 0],
            volcano_coords[excursion_inds, 1], c="r", alpha=0.07,
            s=60)
    for i, ind in enumerate(niklas_data_inds):
        plt.scatter(data_coords[ind, 0], data_coords[ind, 1],
                c=IVRs[i], cmap="jet", marker="*", s=30,
                vmin=ivrs_min, vmax=ivrs_max)
    plt.colorbar()

    plt.savefig("ivrs_nonpaths.png", bbox_inches='tight', pad_inches=0, dpi=400)
    plt.close()
    # plt.show()

    np.save("weighted_IVRs.npy", IVRs)

    """
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
