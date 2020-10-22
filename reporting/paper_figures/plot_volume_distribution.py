""" Plot results of run myopic excursion set reconstruction strategy using weighted IVR.

"""
import os
import torch
import numpy as np
import pandas as pd
from volcapy.grid.grid_from_dsm import Grid
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud, _array_to_point_cloud
from volcapy.plotting.vtkutils import array_to_vector_cloud
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
plot_params = {
        'font.size': 12, 'font.style': 'oblique',
        'axes.labelsize': 'small',
        'axes.titlesize':'small',
        'legend.fontsize': 'small'
        }
plt.rcParams.update(plot_params)


from timeit import default_timer as timer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sample_nr = 4


data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"

static_results_folder = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/static/"



def main(sample_nr):

    results_folder_wIVR = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/wIVR_results/sample_{}/".format(sample_nr)
    
    ground_truth_path = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/post_samples/post_sample_{}.npy".format(sample_nr)

    # Load
    F = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_full_surface.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()

    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"surface_data_coords.npy"))).float()

    ground_truth = torch.from_numpy(np.load(ground_truth_path))

    # Dictionary between the original Niklas data and our discretization.
    niklas_data_inds = torch.from_numpy(
            np.load(os.path.join(data_folder, "niklas_data_inds_insurf.npy"))).long()
    niklas_coords = data_coords[niklas_data_inds].numpy()


    # --------------------------------
    # DEFINITION OF THE EXCURSION SET.
    # --------------------------------
    THRESHOLD_low = 700.0
    excursion_inds = (ground_truth >= THRESHOLD_low).nonzero()[:, 0]
    excu_size = excursion_inds.shape[0] + 1

    excu_sizes_wIVR = []
    for i in range(200, 400):
        cond_real = np.load(
                os.path.join(
                        results_folder_wIVR,
                        "Cond_Reals/conditional_real_{}.npy".format(i)))
        excu_sizes_wIVR.append(np.sum(cond_real >= THRESHOLD_low))

    excu_sizes_interm45 = []
    for i in range(200, 400):
        cond_real = np.load(
                os.path.join(
                        results_folder_wIVR,
                        "Cond_Reals_Interm45/conditional_real_{}.npy".format(i)))
        excu_sizes_interm45.append(np.sum(cond_real >= THRESHOLD_low))

    excu_sizes_prior = []
    for i in range(200, 400):
        cond_real = np.load(
                os.path.join(
                        data_folder,
                        "reskrig_samples/prior_sample_{}.npy".format(i)))
        excu_sizes_prior.append(np.sum(cond_real >= THRESHOLD_low))

    excu_sizes_LIMIT = []
    for i in range(200, 360):
        cond_real = np.load(
                os.path.join(
                        results_folder_wIVR,
                        "Cond_Reals_Infill/conditional_real_{}.npy".format(i)))
        excu_sizes_LIMIT.append(np.sum(cond_real >= THRESHOLD_low))

    sequential_colors = sns.color_palette("RdPu", 10)
    sns.set_palette("RdPu")

    plt.hist(excu_sizes_prior, # color="pink",
            alpha=0.6)
    plt.hist(excu_sizes_interm45, # color="hotpink",
            alpha=0.6)
    plt.hist(excu_sizes_wIVR, # color="palevioletred",
            alpha=0.6)
    plt.hist(excu_sizes_LIMIT, # color="mediumvioletred",
            alpha=0.6)

    # Vertical line at true volume.
    plt.axvline(x=excu_size, color="black", linestyle="--")

    plt.legend(["true volume", "prior", "weighted IVR, 45 observations",
    "weighted IVR, 90 observations", "limiting distribution"])
    plt.savefig("volume_histogram", bbox_inches="tight", pad_inches=0.1, dpi=600)
    plt.show()

if __name__ == "__main__":
    main(4)
