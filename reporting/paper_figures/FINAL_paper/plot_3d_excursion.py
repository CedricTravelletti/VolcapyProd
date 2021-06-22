import os
import torch
import numpy as np
import pandas as pd
from volcapy.grid.grid_from_dsm import Grid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns


sns.set()
sns.set_style("white")
plt.rcParams["font.family"] = "Times New Roman"
plot_params = {
        'font.size': 12, 'font.style': 'oblique',
        'axes.labelsize': 'small',
        'axes.titlesize':'small',
        'legend.fontsize': 'small',
        'xtick.labelsize': 'small'
        }
plt.rcParams.update(plot_params)

my_palette = sns.color_palette("RdBu", 4)
# get colormap from seaborn
cmap = ListedColormap(my_palette.as_hex())

data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"
base_results_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/AISTATS_results_v2/FINAL/"
ground_truths_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/AISTATS_results_v2/FINAL/prior_samples_April2021/"


def main():
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = grid.cells
    data_coords_surface = np.load(os.path.join(data_folder,"surface_data_coords.npy"))
    data_coords_niklas = np.load(os.path.join(data_folder,"niklas_data_coords.npy"))

    X1 = data_coords_niklas[:, 0]
    Y1 = data_coords_niklas[:, 1]
    Z1 = data_coords_niklas[:, 2]

    # Center the separations and scale to meters.
    X1_centred = 111*1e3 * (1e-5 * (X1 - np.min(X1)))
    Y1_centred = 111*1e3 * (1e-5 * (Y1 - np.min(Y1)))

    volcano_X = 111*1e3 * (1e-5 * (volcano_coords[:, 0] -
            np.min(volcano_coords[:, 0])))
    volcano_Y = 111*1e3 * (1e-5 * (volcano_coords[:, 1] -
            np.min(volcano_coords[:, 1])))
    volcano_Z = volcano_coords[:, 2]

    X1_centred_b = 111*1e3 * (1e-5 * (X1 - np.min(volcano_coords[:, 0])))
    Y1_centred_b = 111*1e3 * (1e-5 * (Y1 - np.min(volcano_coords[:, 1])))

    # Prepare figure layout.
    plt.rc('xtick',labelsize=5)
    plt.rc('ytick',labelsize=5)
    fig = plt.figure()
    widths = [1, 1]
    heights = [1, 1, 1, 1]
    gs = fig.add_gridspec(ncols=2, nrows=4, width_ratios=widths,
                                      height_ratios=heights)

    f_ax1 = fig.add_subplot(gs[:, 0])
    f_ax2 = fig.add_subplot(gs[1:2, 1])
    f_ax3 = fig.add_subplot(gs[2:3, 1])

    f_ax1.scatter(X1_centred, Y1_centred, c=Z1, cmap=cmap, linewidth=0.2, s=1)
    f_ax1.set_aspect('equal')
    f_ax1.tick_params(axis='both', which='major', pad=0.1)


    f_ax2.scatter(X1_centred, Z1, c=Z1, cmap=cmap, linewidth=0.2, s=1)
    f_ax2.set_aspect(1.5)
    f_ax2.tick_params(axis='both', which='major', pad=0.1)

    f_ax3.scatter(Y1_centred, Z1, c=Z1, cmap=cmap, linewidth=0.2, s=1)
    f_ax3.set_aspect(1.5)
    f_ax3.tick_params(axis='both', which='major', pad=0.1)

    plt.savefig("niklas_data_overview", bbox_inches="tight", pad_inches=0.1, dpi=400)

    # Now plot the excursions.
    threshold_low = 350.0
    ground_truth = torch.from_numpy(
            np.load(os.path.join(ground_truths_folder, "prior_sample_4.npy")))
    true_excursion_inds = (ground_truth >= threshold_low).nonzero()[:, 0]
    true_excursion_coords = volcano_coords[true_excursion_inds]
    true_excu_densities = ground_truth[true_excursion_inds]

    # Prepare figure layout.
    plt.rc('xtick',labelsize=5)
    plt.rc('ytick',labelsize=5)
    fig = plt.figure()
    widths = [1, 1]
    heights = [1, 1, 1, 1]
    gs = fig.add_gridspec(ncols=2, nrows=4, width_ratios=widths,
                                      height_ratios=heights)

    f_ax1 = fig.add_subplot(gs[:, 0])
    f_ax2 = fig.add_subplot(gs[1:2, 1])
    f_ax3 = fig.add_subplot(gs[2:3, 1])

    f_ax1.scatter(volcano_X[true_excursion_inds],
            volcano_Y[true_excursion_inds], c=true_excu_densities,
            cmap=cmap,
            s=2.8,
            marker="h",
            alpha=0.2, edgecolors='none')
    f_ax1.scatter(X1_centred_b, Y1_centred_b, c="black", linewidth=0.2, s=0.3,
            alpha=1.0)

    f_ax1.set_aspect('equal')
    f_ax1.tick_params(axis='both', which='major', pad=0.1)


    f_ax2.scatter(volcano_X[true_excursion_inds],
            volcano_Z[true_excursion_inds], c=true_excu_densities,
            cmap=cmap,
            s=3,
            marker="h",
            alpha=0.2, edgecolors='none')
    f_ax2.scatter(X1_centred_b, Z1, c="black", linewidth=0.2, s=0.1, alpha=1.0)
    f_ax2.set_aspect(1.5)
    f_ax2.tick_params(axis='both', which='major', pad=0.1)

    f_ax3.scatter(volcano_Y[true_excursion_inds],
            volcano_Z[true_excursion_inds], c=true_excu_densities,
            cmap=cmap,
            s=3,
            marker="h",
            alpha=0.2, edgecolors='none')
    f_ax3.scatter(Y1_centred_b, Z1, c="black", linewidth=0.2, s=0.3, alpha=1.0)
    f_ax3.set_aspect(1.5)
    f_ax3.tick_params(axis='both', which='major', pad=0.1)

    plt.savefig("niklas_data_overview", bbox_inches="tight", pad_inches=0.1, dpi=400)


if __name__ == "__main__":
    main()
