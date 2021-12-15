import os
import torch
import numpy as np
import pandas as pd
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


def plot_excu_profile_with_visited_inds(volcano_coords, data_coords_niklas, 
        visited_coords, visited_coords_lookahead, coast_coords,
        ground_truth, threshold, out_file_path):
    """ Plot the x, y and z profile of the excursion set and the points visited
    by the strategy. Also plot Niklas data.

    """
    X_niklas = data_coords_niklas[:, 0]
    Y_niklas = data_coords_niklas[:, 1]
    Z_niklas = data_coords_niklas[:, 2]

    X_visited = visited_coords[:, 0]
    Y_visited = visited_coords[:, 1]
    Z_visited = visited_coords[:, 2]

    X_visited_lookahead = visited_coords_lookahead[:, 0]
    Y_visited_lookahead = visited_coords_lookahead[:, 1]
    Z_visited_lookahead = visited_coords_lookahead[:, 2]

    # List for coloring visited points according to progress.
    visited_colors = np.array(list(range(X_visited.shape[0])))
    visited_colors_lookahead = np.array(list(range(X_visited_lookahead.shape[0])))

    volcano_X = 111*1e3 * (1e-5 * (volcano_coords[:, 0] -
            np.min(volcano_coords[:, 0])))
    volcano_Y = 111*1e3 * (1e-5 * (volcano_coords[:, 1] -
            np.min(volcano_coords[:, 1])))
    volcano_Z = volcano_coords[:, 2]

    # Center compared to volano.
    X_niklas_centred = 111*1e3 * (1e-5 * (X_niklas - np.min(volcano_coords[:, 0])))
    Y_niklas_centred = 111*1e3 * (1e-5 * (Y_niklas - np.min(volcano_coords[:, 1])))

    X_coast_centred = 111*1e3 * (1e-5 * (coast_coords[:, 0] - np.min(volcano_coords[:, 0])))
    Y_coast_centred = 111*1e3 * (1e-5 * (coast_coords[:, 1] - np.min(volcano_coords[:, 1])))

    X_visited_centred = 111*1e3 * (1e-5 * (X_visited - np.min(volcano_coords[:, 0])))
    Y_visited_centred = 111*1e3 * (1e-5 * (Y_visited - np.min(volcano_coords[:, 1])))

    X_visited_centred_lookahead = 111*1e3 * (1e-5 * (
        X_visited_lookahead - np.min(volcano_coords[:, 0])))
    Y_visited_centred_lookahead = 111*1e3 * (1e-5 * (
        Y_visited_lookahead - np.min(volcano_coords[:, 1])))

    # Get volcano bounding box in centred coordinates.
    min_x = 0.0
    max_x = 111*1e3 * (1e-5 * (np.max(volcano_coords[:, 0]) - np.min(volcano_coords[:, 0])))
    min_y = 0.0
    max_y = 111*1e3 * (1e-5 * (np.max(volcano_coords[:, 1]) - np.min(volcano_coords[:, 1])))
    min_z = np.min(volcano_coords[:, 2])
    max_z = np.max(volcano_coords[:, 2])

    true_excursion_inds = (ground_truth >= threshold).nonzero()[:, 0]
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

    gs.update(wspace=0.001)

    f_ax1 = fig.add_subplot(gs[:, 0])
    f_ax2 = fig.add_subplot(gs[1:2, 1])
    f_ax3 = fig.add_subplot(gs[2:3, 1])

    # Excursion set.
    f_ax1.scatter(volcano_X[true_excursion_inds],
            volcano_Y[true_excursion_inds], c=true_excu_densities,
            cmap=cmap,
            s=2.8,
            marker="h",
            alpha=0.15, edgecolors='none')

    """
    f_ax1.scatter(X_niklas_centred, Y_niklas_centred, c="black", linewidth=0.2, s=0.1,
            alpha=1.0)
    """

    # Add the coast
    f_ax1.scatter(X_coast_centred, Y_coast_centred, c="blue", linewidth=0.2, s=0.1,
            alpha=1.0)

    # Visited locations.
    f_ax1.plot(X_visited_centred, Y_visited_centred,
            color='k',
            linewidth=0.1,
            zorder=-1,
            alpha=1.0)
    f_ax1.scatter(X_visited_centred, Y_visited_centred, c=visited_colors,
            cmap='YlOrRd',
            marker="8", edgecolor='k',
            linewidth=0.1,
            s=4.0,
            zorder=1,
            alpha=1.0)

    # Same for lookahead.
    f_ax1.plot(X_visited_centred_lookahead, Y_visited_centred_lookahead,
            color='k',
            linewidth=0.1,
            zorder=-1,
            alpha=1.0)
    f_ax1.scatter(X_visited_centred_lookahead, Y_visited_centred_lookahead,
            c=visited_colors_lookahead,
            cmap='Greys',
            marker="8", edgecolor='k',
            linewidth=0.1,
            s=4.0,
            zorder=1,
            alpha=1.0)

    f_ax1.set_aspect('equal')
    f_ax1.tick_params(axis='both', which='major', pad=0.1)

    f_ax1.set_xlim([min_x, max_x])
    f_ax1.set_ylim([min_y, max_y])


    # Excursion set.
    f_ax2.scatter(volcano_X[true_excursion_inds],
            volcano_Z[true_excursion_inds], c=true_excu_densities,
            cmap=cmap,
            s=3,
            marker="h",
            alpha=0.2, edgecolors='none')
    f_ax2.scatter(X_niklas_centred, Z_niklas, c="black", linewidth=0.2, s=0.05, alpha=1.0)

    # Visited location.
    f_ax2.plot(X_visited_centred, Z_visited,
            color='k',
            linewidth=0.1,
            zorder=-1,
            alpha=1.0)
    f_ax2.scatter(X_visited_centred, Z_visited, c=visited_colors,
            cmap='YlOrRd',
            marker="8", edgecolor='k',
            linewidth=0.1,
            zorder=1,
            s=2.5,
            alpha=1.0)

    # Same for lookahead.
    f_ax2.plot(X_visited_centred_lookahead, Z_visited_lookahead,
            color='k',
            linewidth=0.1,
            zorder=-1,
            alpha=1.0)
    f_ax2.scatter(X_visited_centred_lookahead, Z_visited_lookahead,
            c=visited_colors_lookahead,
            cmap='Greys',
            marker="8", edgecolor='k',
            linewidth=0.1,
            zorder=1,
            s=2.5,
            alpha=1.0)

    f_ax2.set_aspect(1.5)
    f_ax2.tick_params(axis='both', which='major', pad=0.1)

    f_ax2.set_xlim([min_x, max_x])
    f_ax2.set_ylim([min_z, max_z + 280.0])

    # Excursion set.
    f_ax3.scatter(volcano_Y[true_excursion_inds],
            volcano_Z[true_excursion_inds], c=true_excu_densities,
            cmap=cmap,
            s=3,
            marker="h",
            alpha=0.2, edgecolors='none')
    f_ax3.scatter(Y_niklas_centred, Z_niklas, c="black", linewidth=0.2, s=0.05, alpha=1.0)

    # Visited locations.
    f_ax3.plot(Y_visited_centred, Z_visited,
            color='k',
            linewidth=0.1,
            zorder=-1,
            alpha=1.0)
    f_ax3.scatter(Y_visited_centred, Z_visited, c=visited_colors,
            cmap='YlOrRd',
            marker="8", edgecolor='k',
            linewidth=0.1,
            s=2.5,
            zorder=1,
            alpha=1.0)

    # Same for lookahead.
    f_ax3.plot(Y_visited_centred_lookahead, Z_visited_lookahead,
            color='k',
            linewidth=0.1,
            zorder=-1,
            alpha=1.0)
    f_ax3.scatter(Y_visited_centred_lookahead, Z_visited_lookahead,
            c=visited_colors_lookahead,
            cmap='Greys',
            marker="8", edgecolor='k',
            linewidth=0.1,
            s=2.5,
            zorder=1,
            alpha=1.0)

    f_ax3.set_aspect(1.5)
    f_ax3.tick_params(axis='both', which='major', pad=0.1)

    f_ax3.set_xlim([min_y, max_y])
    f_ax3.set_ylim([min_z, max_z + 280.0])

    plt.savefig(out_file_path, bbox_inches="tight", pad_inches=0.1, dpi=400)
