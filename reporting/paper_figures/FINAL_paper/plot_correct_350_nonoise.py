""" Plot results of run myopic excursion set reconstruction strategy using weighted IVR.

"""
import os
import torch
import numpy as np
import pandas as pd
import volcapy.covariance.exponential as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance import UpdatableGP
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plot_params = {
        'font.size': 18, 'font.style': 'oblique',
        'axes.labelsize': 'small',
        'axes.titlesize':'small',
        'legend.fontsize': 'small'
        }
plt.rcParams.update(plot_params)

data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"
base_results_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/AISTATS_results_v2/FINAL"

def process_sample(sample_nr):
    results_folder_IVR = os.path.join(base_results_folder,
            "IVR_results/prior_samples_April2021/sample_{}/".format(sample_nr))
    results_folder_wIVR = os.path.join(base_results_folder,
            "wIVR_results_350_nonoise_step_310/prior_samples_April2021/sample_{}/".format(sample_nr))
    
    results_folder_INFILL = os.path.join(base_results_folder,
            "INFILL_results_700_nonoise/prior_samples_April2021/sample_{}/".format(sample_nr))
    
    ground_truth_path = os.path.join(base_results_folder,
            "prior_samples_April2021/prior_sample_{}.npy".format(sample_nr))


    # Load
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()

    ground_truth = torch.from_numpy(np.load(ground_truth_path))

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

    visited_inds_IVR = visited_inds_IVR[:90]
    visited_inds_wIVR = visited_inds_wIVR[:90]


    def compute_mismatch(coverage):
        # -------------------------------------------------
        # -------------------------------------------------
        # Find the Vorob'ev threshold.
        vorobev_volume = np.sum(coverage)
        def f(x):
            return np.sum(coverage > x) - vorobev_volume

        from scipy.optimize import brentq
        vorobev_threshold = brentq(f, 0.0, 1.0)
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

    def compute_vorobev(coverage):
        # -------------------------------------------------
        # -------------------------------------------------
        # Find the Vorob'ev threshold.
        vorobev_volume = np.sum(coverage)
        def f(x):
            return np.sum(coverage > x) - vorobev_volume

        from scipy.optimize import brentq
        vorobev_threshold = brentq(f, 0.0, 1.0)
        print("Vorob'ev threshold: {}.".format(vorobev_threshold))

        # Plot estimated excursion set using coverage function.
        est_excursion_inds = coverage > vorobev_threshold

        est_excursion = np.zeros(volcano_coords.shape[0])
        est_excursion[est_excursion_inds] = 1
        return est_excursion

    mismatches_IVR = []
    mismatches_wIVR = []
    mismatches_INFILL = []

    # Compute the mismatch evolution for all strategies.
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


    # Save the last estimate.
    # vorobev_est_INFILL = compute_vorobev(coverage_INFILL)
    vorobev_est_IVR = compute_vorobev(coverage_IVR)
    vorobev_est_wIVR = compute_vorobev(coverage_wIVR)

    irregular_array_to_point_cloud(
            volcano_coords.numpy(),
            vorobev_est_IVR,
            os.path.join(results_folder_IVR, "vorobev_est.vtk"),
            fill_nan_val=-20000.0)
    irregular_array_to_point_cloud(
            volcano_coords.numpy(),
            vorobev_est_wIVR,
            os.path.join(results_folder_wIVR, "vorobev_est.vtk"),
            fill_nan_val=-20000.0)

    mismatches_IVR = np.array(mismatches_IVR)
    mismatches_wIVR = np.array(mismatches_wIVR)

    X_IVR = np.array(list(range(1, visited_inds_IVR.shape[0])))
    X_wIVR = np.array(list(range(1, visited_inds_wIVR.shape[0])))

    # Put an horizontal line for the limiting distribution.
    coverage_INFILL = np.load(
                os.path.join(
                        results_folder_INFILL,
                        "coverage_99.npy"))
    mismatches_INFILL = np.array(compute_mismatch(coverage_INFILL))

    n = 600
    X_INFILL_last = np.array(list(range(1, n)))
    mismatches_INFILL_last = np.repeat(mismatches_INFILL[:].reshape(1, -1), n-1,
            axis=0)

    df1 = pd.DataFrame({'X': X_IVR, 'false positives': mismatches_IVR[:, 0],
            'false negatives': mismatches_IVR[:, 1], 'correct': mismatches_IVR[:,2],
            'strategy': 'IVR, long range'})

    df2 = pd.DataFrame({'X': X_wIVR, 'false positives': mismatches_wIVR[:, 0],
            'false negatives': mismatches_wIVR[:, 1], 'correct':
            mismatches_wIVR[:,2],
            'strategy': 'weighted IVR, long range'})

    df5 = pd.DataFrame({'X': X_INFILL_last, 'false positives':
            mismatches_INFILL_last[:, 0],
            'false negatives': mismatches_INFILL_last[:, 1], 'correct':
            mismatches_INFILL_last[:,2],
            'strategy': 'limiting distribution'})

    df = pd.concat([df1, df2, df5])
    # df = pd.concat([df1, df2, df3, df5])

    return df

if __name__ == "__main__":
    df = process_sample(4)
    for sample_nr in range(5,9):
        tmp = process_sample(sample_nr)
        df = pd.concat([df, tmp])
    sns.lineplot(data=df, x='X', y='correct', hue='strategy', style='strategy',
            n_boot=10000, legend=False)
    plt.legend(loc='upper left',
        labels=['IVR, long range', 'weighted IVR, long range',
                'limiting distribution'])
    plt.xlim([1, 90]) 

    plt.xlabel("Number of observations")
    plt.ylabel("True positives in percent of true volume.")

    plt.savefig("mismatch_evolution_correct_350_nonoise_step_310", bbox_inches="tight", pad_inches=0.1, dpi=600)
