""" Plot histogram of prior excursion volume.

"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


static_data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"
base_results_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/AISTATS_results_v2/FINAL/"


def main():
    thresholds = [500.0, 800.0, 1000, 1250]
    df = pd.DataFrame()
    for threshold in thresholds:
        excursion_sizes = []
        for i in range(1, 400):
            ground_truth_path = os.path.join(base_results_folder,
                    "final_samples_matern32/prior_sample_{}.npy".format(i))
            ground_truth = torch.from_numpy(np.load(ground_truth_path))
    
            true_excursion_inds = (ground_truth >= threshold).nonzero()[:, 0]
            excursion_size = 100 * (true_excursion_inds.shape[0] / ground_truth.shape[0])
            df = df.append({'Excursion size (percent of total volume)': excursion_size,
                    'threshold': threshold, 'sample_nr': i}, ignore_index=True)
    # my_palette = sns.color_palette("rocket", 4)
    my_palette = sns.color_palette("RdBu", 4)

    sns.histplot(df, x='Excursion size (percent of total volume)',
            kde=True, binwidth=2,
            hue='threshold', palette=my_palette)
    plt.savefig("histogram", bbox_inches="tight", pad_inches=0.1, dpi=600)

    # Now plot distribution of volume for small and big threshold.
    threshold_small = 1250.0
    threshold_big = 1000.0

    # Palette defining the colors of the samples.
    samples_palette = sns.color_palette("RdBu", 8)
    samples_palette = samples_palette[:3] + samples_palette[-2:]

    # Replot the histogram, but only for the big and small thresholds.
    sns.histplot(
        df[(df["threshold"] == threshold_small)], x='Excursion size (percent of total volume)',
            kde=True, binwidth=2,
            hue='threshold', palette=[my_palette[0]],
            alpha=0.25)

    # Add vertical lines for each of the considered samples.
    line_handles = []
    for i in range(1, 6):
        vol = df.loc[(df['sample_nr'] == i) & (df['threshold'] == threshold_small) , "Excursion size (percent of total volume)"].item()
        line_handles.append(plt.axvline(x=vol, linewidth=3, linestyle='--',
                label="sample {}".format(i),
                color = samples_palette[i-1]
                ))
    plt.legend(handles=line_handles)
    plt.savefig("histogram_with_small_volumes", bbox_inches="tight", pad_inches=0.1, dpi=600)
    plt.show()

    # SAME FOR BIG
    sns.histplot(
        df[(df["threshold"] == threshold_big)], x='Excursion size (percent of total volume)',
            kde=True, binwidth=2,
            hue='threshold', palette=[my_palette[-1]],
            alpha=0.25)

    # Add vertical lines for each of the considered samples.
    line_handles = []
    for i in range(1, 6):
        vol = df.loc[(df['sample_nr'] == i) & (df['threshold'] == threshold_big) , "Excursion size (percent of total volume)"].item()
        line_handles.append(plt.axvline(x=vol, linewidth=3, linestyle='--',
                label="sample {}".format(i),
                color = samples_palette[i-1]
                ))
    plt.legend(handles=line_handles)
    plt.savefig("histogram_with_big_volumes", bbox_inches="tight", pad_inches=0.1, dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
