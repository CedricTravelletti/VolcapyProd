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
plt.rcParams["font.family"] = "serif"
plot_params = {
        'font.size': 20, 'font.style': 'normal',
        'axes.labelsize': 'small',
        'axes.titlesize':'small',
        'legend.fontsize': 'small',
        'legend.title_fontsize': 'small'
        }
plt.rcParams.update(plot_params)


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
    results_folder = os.path.join(base_results_folder,
            "wIVR_final_big/")
    infill_folder = os.path.join(base_results_folder,
            "INFILL_final_big/")

    situations_folders = [results_folder, infill_folder]
    strategies = ["wIVR, big set", "limiting distribution"]

    thresholds = [2500.0, 2500.0
            ]
    n_datapoints_list = [list(range(0, 451)), list(range(0, 190, 10))]
    df = pd.DataFrame()
    for (situation, strategy, threshold, n_datapoints) in zip(
            situations_folders, strategies, thresholds, n_datapoints_list):
        print(situation)
        for sample_nr in range(1, 6):

            ground_truth_path = os.path.join(
                    os.path.join(base_results_folder,
                    "final_samples_matern32/"),
                    "prior_sample_{}.npy".format(sample_nr))

            coverages_folder = os.path.join(
                    os.path.join(base_results_folder, situation),
                    "sample_{}/".format(sample_nr))
            df = pd.concat(
                    [df, process_sample(
                ground_truth_path, coverages_folder, sample_nr,
                n_datapoints, strategy, threshold)])


df['n_datapoints'] = df.index
df['n_datapoints'][df['strategy'] == "limiting distribution"] = (5 * 
        df['n_datapoints'][df['strategy'] == "limiting distribution"])

my_palette = sns.color_palette("RdBu", 8)
my_palette = my_palette[:3] + my_palette[-2:]

ax = sns.lineplot('n_datapoints', 'correct', ci=None, hue='sample_nr',
        style="strategy",
        data=df, palette=my_palette)
plt.setp(ax.lines[1], alpha=.4)
plt.setp(ax.lines[3], alpha=.4)
plt.setp(ax.lines[5], alpha=.4)
plt.setp(ax.lines[7], alpha=.4)
plt.setp(ax.lines[9], alpha=.4)

ax.set_xlim([-2, 455])
ax.set_xlabel("Number of observations")
ax.set_ylabel("True positives [% of true excursion volume]")
plt.savefig("final_big_detection", bbox_inches="tight", pad_inches=0.1, dpi=400)
plt.show()

ax = sns.lineplot('n_datapoints', 'false positives', ci=None, hue='sample_nr',
        style="strategy",
        data=df, palette=my_palette)
plt.setp(ax.lines[1], alpha=.4)
plt.setp(ax.lines[3], alpha=.4)
plt.setp(ax.lines[5], alpha=.4)
plt.setp(ax.lines[7], alpha=.4)
plt.setp(ax.lines[9], alpha=.4)

ax.set_xlim([-2, 455])
ax.set_xlabel("Number of observations")
ax.set_ylabel("False positives [% of complementary of true excursion volume]")
plt.savefig("final_big_falsepos", bbox_inches="tight", pad_inches=0.1, dpi=400)
plt.show()
