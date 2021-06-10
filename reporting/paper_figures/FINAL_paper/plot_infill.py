""" Plot results of run myopic excursion set reconstruction strategy using weighted IVR.

"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from volcapy.plotting.plot_helper_paper import compute_mismatches


sns.set()
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
plot_params = {
        'font.size': 16, 'font.style': 'oblique',
        'axes.labelsize': 'small',
        'axes.titlesize':'small',
        'legend.fontsize': 'small'
        }
plt.rcParams.update(plot_params)

static_data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"
base_results_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/AISTATS_results_v2/FINAL/"
results_folder_INFILL = os.path.join(base_results_folder,
        "INFILL_results_350_nonoise/prior_samples_April2021/")
threshold_low = 350.0

n_data_points = list(range(0, 1210, 10))


def process_sample(sample_nr):
    ground_truth_path = os.path.join(base_results_folder,
            "prior_samples_April2021/prior_sample_{}.npy".format(sample_nr))
    ground_truth = torch.from_numpy(np.load(ground_truth_path))

    # Load coverages.
    coverages = []
    for i in n_data_points:
        coverages.append(np.load(os.path.join(
                os.path.join(results_folder_INFILL, "sample_{}/".format(sample_nr)),
                "coverage_{}.npy".format(i))))
    df = compute_mismatches(ground_truth, coverages, n_data_points,
            threshold_low, static_data_folder)
    # Add label.
    df['sample_nr'] = sample_nr

    return df

if __name__ == "__main__":
    df = process_sample(4)
    for sample_nr in range(5,9):
        tmp = process_sample(sample_nr)
        df = pd.concat([df, tmp])

    sns.lineplot(data=df, x='n_datapoints', y='correct', hue='sample_nr',
            style='sample_nr',
            legend=True)
    plt.savefig("correct_evolution_infill_350_nonoise", bbox_inches="tight", pad_inches=0.1, dpi=600)
    plt.show()
