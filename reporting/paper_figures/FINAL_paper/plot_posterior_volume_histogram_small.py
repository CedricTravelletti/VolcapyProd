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

threshold_small = 2600.0


def main():
    df = pd.DataFrame()
    true_excursion_sizes = []
    for sample_nr in range(1, 6):
        ground_truth_path = os.path.join(base_results_folder,
                "final_samples_matern32/prior_sample_{}.npy".format(sample_nr))
        ground_truth = torch.from_numpy(np.load(ground_truth_path))

        true_excursion_inds = (ground_truth >= threshold_small).nonzero()[:, 0]
        excursion_size = 100 * (true_excursion_inds.shape[0] / ground_truth.shape[0])

        true_excursion_sizes.append(excursion_size)

        for i in range(11, 400):
            sample_path = os.path.join(base_results_folder,
                    "wIVR_final_small/sample_{}/post_samples/post_sample_{}.npy".format(sample_nr, i))
            sample = torch.from_numpy(np.load(sample_path))
            excursion_inds = (sample >= threshold_small).nonzero()[:, 0]
            excursion_size = 100 * (excursion_inds.shape[0] / ground_truth.shape[0])
            df = df.append({'Excursion size (percent of total volume)': excursion_size,
                    'sample_nr': sample_nr, 'post_sample_nr': i}, ignore_index=True)

    
    my_palette = sns.color_palette("RdBu", 8)
    my_palette = my_palette[:3] + my_palette[-2:]

    sns.histplot(df, x='Excursion size (percent of total volume)',
            kde=True, binwidth=0.5,
            hue='sample_nr', palette=my_palette)

    # Add vertical lines for each of the considered samples.
    line_handles = []
    for i in range(1, 6):
        line_handles.append(plt.axvline(x=true_excursion_sizes[i-1], linewidth=3, linestyle='--',
                label="sample {}".format(i),
                color = my_palette[i-1]
                ))
    plt.legend(handles=line_handles)

    plt.savefig("histogram_posterior_small", bbox_inches="tight", pad_inches=0.1, dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
