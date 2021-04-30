""" Plot visited indices on top of excursion set.

"""
import os
import torch
import numpy as np
import pandas as pd
import volcapy.covariance.exponential as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance import UpdatableGP
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud, _array_to_point_cloud
from volcapy.plotting.vtkutils import array_to_vector_cloud
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plot_params = {
        'font.size': 12, 'font.style': 'oblique',
        'axes.labelsize': 'small',
        'axes.titlesize':'small',
        'legend.fontsize': 'small'
        }
plt.rcParams.update(plot_params)

data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"
base_results_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/AISTATS_results_v2/FINAL/"


def process_sample(sample_nr):
    results_folder_IVR = os.path.join(base_results_folder,
            "IVR_results/prior_samples_April2021/sample_{}/".format(sample_nr))
    results_folder_wIVR = os.path.join(base_results_folder,
            "wIVR_results_700_nonoise/prior_samples_April2021/sample_{}/".format(sample_nr))
    
    results_folder_INFILL = os.path.join(base_results_folder,
            "INFILL_results_700_nonoise/prior_samples_April2021/sample_{}/".format(sample_nr))
    
    ground_truth_path = os.path.join(base_results_folder,
            "prior_samples_April2021/prior_sample_{}.npy".format(sample_nr))


    # Load
    F = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_full_surface.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()
    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"surface_data_coords.npy"))).float()
    ground_truth = torch.from_numpy(np.load(ground_truth_path))

    # --------------------------------
    # DEFINITION OF THE EXCURSION SET.
    # --------------------------------
    THRESHOLD_low = 700.0
    excursion_inds = (ground_truth >= THRESHOLD_low).nonzero()[:, 0]

    # Load results.
    visited_inds_wIVR = np.load(os.path.join(
            results_folder_wIVR, "visited_inds.npy"))

    visited_inds_wIVR = visited_inds_wIVR[:90]
    wIVR_coords = data_coords[visited_inds_wIVR]

    plt.scatter(
            volcano_coords[excursion_inds, 0],
            volcano_coords[excursion_inds, 1], c="r", alpha=0.07)
    plt.scatter(
            wIVR_coords[:, 0],
            wIVR_coords[:, 1], c="k", s=4)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    plt.savefig("wIVR_visited_{}.png".format(sample_nr),
            bbox_inches="tight", pad_inches=0, dpi=400)
    plt.show()


if __name__ == "__main__":
    df = process_sample(4)
