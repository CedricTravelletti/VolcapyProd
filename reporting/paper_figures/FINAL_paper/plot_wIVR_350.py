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
        "wIVR_results_350_nonoise_step_310/prior_samples_April2021/")
threshold_low = 350.0


def process_sample(sample_nr, n_datapoints):
    ground_truth_path = os.path.join(base_results_folder,
            "prior_samples_April2021/prior_sample_{}.npy".format(sample_nr))
    ground_truth = torch.from_numpy(np.load(ground_truth_path))

    # Load coverages.
    coverages = []
    for i in n_datapoints:
        coverages.append(np.load(os.path.join(
                os.path.join(results_folder_INFILL, "sample_{}/".format(sample_nr)),
                "coverage_{}.npy".format(i))))
    df = compute_mismatches(ground_truth, coverages, n_datapoints,
            threshold_low, static_data_folder)
    # Add label.
    df['sample_nr'] = sample_nr

    return df

if __name__ == "__main__":
    n_datapoints_4 = list(range(0, 35))
    n_datapoints_5 = list(range(0, 49))
    n_datapoints_6 = list(range(0, 19))
    n_datapoints_7 = list(range(0, 19))
    n_datapoints_8 = list(range(0, 19))

    df4 = process_sample(4, n_datapoints_4)
    df5 = process_sample(5, n_datapoints_5)
    df6 = process_sample(6, n_datapoints_6)
    df7 = process_sample(7, n_datapoints_7)
    df8 = process_sample(8, n_datapoints_8)
    df = pd.concat([df4, df5, df6, df7, df8])

    sns.lineplot(data=df, x='n_datapoints', y='correct', hue='sample_nr',
            style='sample_nr',
            legend=True)
    plt.savefig("correct_evolution_wIVR_350_nonoise_step_310", bbox_inches="tight", pad_inches=0.1, dpi=600)
    plt.show()
