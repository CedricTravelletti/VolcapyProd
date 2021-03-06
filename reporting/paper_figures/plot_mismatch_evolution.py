""" Plot results of run myopic excursion set reconstruction strategy using weighted IVR.

"""
import os
import torch
import numpy as np
import volcapy.covariance.exponential as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance import UpdatableGP
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
        'legend.fontsize': 'xx-small'
        }
plt.rcParams.update(plot_params)


from timeit import default_timer as timer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sample_nr = 5


data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"
results_folder_IVR = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/IVR_results/sample_{}/".format(sample_nr)
results_folder_wIVR =
"/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/wIVR_results/sample_{}/".format(sample_nr)

infill_folder = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/INFILL_results/sample_{}/".format(sample_nr)

static_results_folder = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/static/"

ground_truth_path = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/post_samples/post_sample_{}.npy".format(sample_nr)

# Indices of the data points that are along the coast.
from volcapy.data_preparation.paths import coast_data_inds


def main():

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

    # Load results.
    visited_inds_IVR = np.load(os.path.join(
            results_folder_IVR, "visited_inds.npy"))
    visited_inds_wIVR = np.load(os.path.join(
            results_folder_wIVR, "visited_inds.npy"))

    def compute_mismatch(coverage):
        # -------------------------------------------------
        # -------------------------------------------------
        # Find the Vorob'ev threshold.
        vorobev_volume = np.sum(coverage)
        def f(x):
            return (np.abs(np.sum(coverage > x) - vorobev_volume))

        from scipy.optimize import minimize
        vorobev_threshold = minimize(f, 0.5, method='nelder-mead',
                               options={'xatol': 1e-3, 'disp': True}).x
        print("Vorob'ev threshold: {}.".format(vorobev_threshold))

        # Plot estimated excursion set using coverage function.
        est_excursion_inds = coverage > vorobev_threshold

        print("Vorobe'v volume: {}.".format(vorobev_volume))
        print("Current estimate volume: {}.".format(np.sum(est_excursion_inds)))

        # Compute mismatch.
        mismatch = np.zeros(volcano_coords.shape[0])
        mismatch[est_excursion_inds] = 1
        tmp = np.zeros(volcano_coords.shape[0])
        tmp[excursion_inds] = 2
        mismatch = mismatch + tmp

        excu_size = excursion_inds.shape[0] + 1
        p_false_pos = 100 * np.sum(mismatch == 1) / excu_size
        p_false_neg = 100 * np.sum(mismatch == 2) / excu_size
        p_correct = 100 * np.sum(mismatch == 3) / excu_size

        return (p_false_pos, p_false_neg, p_correct)

    mismatches_IVR = []
    mismatches_wIVR = []
    mismatches_static = []

    # Number of static datapoints.
    n_static = 542
    for i in range(1, n_static):
        coverage_static = np.load(
                os.path.join(
                        static_results_folder,
                        "coverage_{}.npy".format(i)))
        mismatches_static.append(compute_mismatch(coverage_static))

    for i in range(1, visited_inds_IVR.shape[0]):
        coverage_IVR = np.load(
                os.path.join(
                        results_folder_IVR,
                        "coverage_{}.npy".format(i)))
        mismatches_IVR.append(compute_mismatch(coverage_IVR))

    for i in range(1, visited_inds_wIVR.shape[0]):
        coverage_wIVR = np.load(
                os.path.join(
                        results_folder_wIVR,
                        "coverage_{}.npy".format(i)))
        mismatches_wIVR.append(compute_mismatch(coverage_wIVR))

    mismatches_IVR = np.array(mismatches_IVR)
    mismatches_wIVR = np.array(mismatches_wIVR)
    mismatches_static = np.array(mismatches_static)

    plt.plot(list(range(1, visited_inds_IVR.shape[0])), mismatches_IVR[:,0],
            label="false positives, IVR strategy",
            color="blue", linestyle="dashed")
    plt.plot(list(range(1, visited_inds_IVR.shape[0])), mismatches_IVR[:,1],
            label="false negatives, IVR strategy",
            color="red", linestyle="dashed")
    plt.plot(list(range(1, visited_inds_IVR.shape[0])), mismatches_IVR[:,2],
            label="correct prediction, IVR strategy",
            color="green", linestyle="dashed")

    plt.plot(list(range(1, visited_inds_wIVR.shape[0])), mismatches_wIVR[:,0],
            label="false positives, weighted IVR strategy",
            color="cornflowerblue", linestyle="solid")
    plt.plot(list(range(1, visited_inds_wIVR.shape[0])), mismatches_wIVR[:,1],
            label="false negatives, weighted IVR strategy",
            color="lightcoral", linestyle="solid")
    plt.plot(list(range(1, visited_inds_wIVR.shape[0])), mismatches_wIVR[:,2],
            label="correct prediction, weighted IVR strategy",
            color="lightgreen", linestyle="solid") 

    # Add the surface fill coverage.
    coverage_fill = np.load(os.path.join(
            infill_folder, "coverage_surface_fill.npy"))
    mismatch_fill = compute_mismatch(coverage_fill)

    # Size of the limiting horizontal line corresponding to the infill
    # strategy.
    n = 600

    plt.plot(list(range(1, n)), np.repeat(mismatch_fill[0], n-1),
            label="false positives, surface fill",
            color="cornflowerblue", linestyle="dotted")
    plt.plot(list(range(1, n)), np.repeat(mismatch_fill[1], n-1),
            label="false negatives, surface fill",
            color="lightcoral", linestyle="dotted")
    plt.plot(list(range(1, n)), np.repeat(mismatch_fill[2], n-1),
            label="correct prediction, surface fill",
            color="lightgreen", linestyle="dotted") 
    
    plt.xlim([1, 500]) 
    plt.legend()
    plt.xlabel("Number of observations")
    plt.ylabel("Size in percent of true excursion size")
    plt.savefig("mismatch_evolution_bigstep", bbox_inches="tight", pad_inches=0.1, dpi=600)
    plt.show()

if __name__ == "__main__":
    main()
