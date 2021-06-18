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
sns.set_style("white")
plt.rcParams["font.family"] = "Times New Roman"
plot_params = {
        'font.size': 16, 'font.style': 'oblique',
        'axes.labelsize': 'small',
        'axes.titlesize':'small',
        'legend.fontsize': 'small'
        }
plt.rcParams.update(plot_params)

my_palette = sns.color_palette("RdBu", 4)

static_data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"
base_results_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/AISTATS_results_v2/FINAL/"


def process_sample(
        ground_truth_path, coverages_folder, sample_nr, n_datapoints, strategy, threshold):
    ground_truth = torch.from_numpy(np.load(ground_truth_path))

    # Load coverages.
    coverages = []
    for i in n_datapoints:
        coverages.append(np.load(
                os.path.join(coverages_folder, "coverage_{}.npy".format(i))))
    df = compute_mismatches(ground_truth, coverages, n_datapoints,
            threshold, static_data_folder)
    # Add label.
    df = df.set_index('n_datapoints', drop=True)
    df['sample_nr'] = sample_nr
    df['threshold'] = threshold
    df['strategy'] = strategy

    # For limiting distribution, only put last value.
    if strategy == 'limiting distribution':
        df['false positives'] = df.iloc[-1]['false positives']
        df['false negatives'] = df.iloc[-1]['false negatives']
        df['correct'] = df.iloc[-1]['correct']

    return df

if __name__ == "__main__":
    threshold_low = 350.0
    threshold_high = 700.0
    
    results_folder_INFILL_low = os.path.join(base_results_folder,
            "INFILL_results_350_nonoise/prior_samples_April2021/")
    results_folder_INFILL_high = os.path.join(base_results_folder,
            "INFILL_results_700_nonoise/prior_samples_April2021/")

    results_folder_wIVR_310_low = os.path.join(base_results_folder,
            "wIVR_results_350_nonoise_step_310/prior_samples_April2021/")
    results_folder_wIVR_600_low = os.path.join(base_results_folder,
            "wIVR_results_350_nonoise_step_600/prior_samples_April2021/")

    results_folder_wIVR_310_high = os.path.join(base_results_folder,
            "wIVR_results_700_nonoise_step_310/prior_samples_April2021/")
    results_folder_wIVR_600_high = os.path.join(base_results_folder,
            "wIVR_results_700_nonoise_step_600/prior_samples_April2021/")

    results_folder_IVR_310_low = os.path.join(base_results_folder,
            "IVR_results_350_nonoise_step_310/prior_samples_April2021/")
    results_folder_IVR_600_low = os.path.join(base_results_folder,
            "IVR_results_350_nonoise_step_600/prior_samples_April2021/")

    results_folder_IVR_310_low = os.path.join(base_results_folder,
            "IVR_results_700_nonoise_step_310/prior_samples_April2021/")
    results_folder_IVR_600_low = os.path.join(base_results_folder,
            "IVR_results_700_nonoise_step_600/prior_samples_April2021/")

    situations_folders = [results_folder_INFILL_low, results_folder_INFILL_high,
            results_folder_wIVR_310_low, results_folder_wIVR_600_low,
            results_folder_wIVR_310_high, results_folder_wIVR_600_high,]
    strategies = ["limiting distribution", "limiting distribution",
            "wIVR, max step: 310 [m]", "wIVR, max step: 600 [m]",
            "wIVR, max step: 310 [m]", "wIVR, max step: 600 [m]",]

    thresholds = [350.0, 700.0,
            350.0, 350.0,
            700.0, 700.0,
            ]
    n_datapoints_list = [list(range(0, 1210, 10)), list(range(0, 1210, 10)),
        list(range(0, 19)), list(range(0, 19)),
        list(range(0, 19)), list(range(0, 19)),]
    df = pd.DataFrame()
    for (situation, strategy, threshold, n_datapoints) in zip(
            situations_folders, strategies, thresholds, n_datapoints_list):
        print(situation)
        for sample_nr in range(4, 9):

            ground_truth_path = os.path.join(
                    os.path.join(base_results_folder,
                    "prior_samples_April2021/"),
                    "prior_sample_{}.npy".format(sample_nr))

            coverages_folder = os.path.join(
                    os.path.join(base_results_folder, situation),
                    "sample_{}/".format(sample_nr))
            df = pd.concat(
                    [df, process_sample(
                ground_truth_path, coverages_folder, sample_nr,
                n_datapoints, strategy, threshold)])

my_palette = sns.color_palette("RdBu", 8)
my_palette = my_palette[:3] + my_palette[-2:]
ax = sns.lineplot('n_datapoints', 'correct', ci=None, hue='sample_nr',
        style="strategy",
        data=df, palette=my_palette)

ax.set_xlim([0, 30])
ax.set_xlabel("Number of observations")
ax.set_ylabel("Detection percentage [% of true excursion volume]")
plt.savefig("correct_detection_evolution", bbox_inches="tight", pad_inches=0.1, dpi=400)
plt.show()
