""" Plot results of run myopic excursion set reconstruction strategy using weighted IVR.

"""
import os
import torch
import numpy as np
import pandas as pd
from volcapy.uq.set_estimation import vorobev_expectation_inds, vorobev_quantile_inds
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


def compute_mismatches(ground_truth, coverages, n_datapoints, threshold_low, static_data_folder):
    """ Given a list of coverages (at different sampling steps), compute the
    mismatch between the estimated excursion set (using Vorob'ev expectation)
    and the true excursion set.

    Parameters
    ----------
    ground_truth: (n_cells) array-like
        True value of the field at each cell.
    coverages: List[(n_cells) array-like]
        List of different estimations of the excursion probability at each cell.
    n_datapoints: List[int]
        List containing the number of datapoints that was used for each
        estimation.
    threshold_low: float
        Threshold (lower) defining the excursion set.
    static_data_folder: string
        Path to folder containing static data for the volcano (discretization
        grid, ...).

    Returns
    -------
    pandas DataFrame
        ['n_datapoints', 'false positives', 'false negatives','correct']
        n_datapoints: List[int]: number of datapoints ingested at step n.
        false_positives: number of false positives as percentage of true
        excursion size.

        Same for the others.

    """
    # Load static data.
    grid = Grid.load(os.path.join(static_data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()

    true_excursion_inds = (ground_truth >= threshold_low).nonzero()[:, 0]

    def compute_mismatch(coverage):
        # Plot estimated excursion set using coverage function.
        est_excursion_inds = vorobev_expectation_inds(coverage)

        # Compute mismatch.
        mismatch = np.zeros(volcano_coords.shape[0])
        mismatch[est_excursion_inds] = 1
        tmp = np.zeros(volcano_coords.shape[0])
        tmp[true_excursion_inds] = 2
        mismatch = mismatch + tmp

        excu_size = true_excursion_inds.shape[0] + 1
        p_false_pos = 100 * np.sum(mismatch == 1) / excu_size
        p_false_neg = 100 * np.sum(mismatch == 2) / excu_size
        p_correct = 100 * np.sum(mismatch == 3) / excu_size

        return (p_false_pos, p_false_neg, p_correct)

    mismatches = []
    # Compute the mismatch evolution.
    for coverage in coverages:
        mismatches.append(compute_mismatch(coverage))
    mismatches = np.array(mismatches)

    df = pd.DataFrame({'n_datapoints': n_datapoints, 'false positives': mismatches[:, 0],
            'false negatives': mismatches[:, 1], 'correct': mismatches[:,2],
            })
    return df

def compute_mismatches_quantile(ground_truth, quantile, coverages, n_datapoints, threshold_low, static_data_folder):
    """ Given a list of coverages (at different sampling steps), compute the
    mismatch between the estimated excursion set (using Vorob'ev expectation)
    and the true excursion set.

    Parameters
    ----------
    ground_truth: (n_cells) array-like
        True value of the field at each cell.
    coverages: List[(n_cells) array-like]
        List of different estimations of the excursion probability at each cell.
    n_datapoints: List[int]
        List containing the number of datapoints that was used for each
        estimation.
    threshold_low: float
        Threshold (lower) defining the excursion set.
    static_data_folder: string
        Path to folder containing static data for the volcano (discretization
        grid, ...).

    Returns
    -------
    pandas DataFrame
        ['n_datapoints', 'false positives', 'false negatives','correct']
        n_datapoints: List[int]: number of datapoints ingested at step n.
        false_positives: number of false positives as percentage of true
        excursion size.

        Same for the others.

    """
    # Load static data.
    grid = Grid.load(os.path.join(static_data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()

    true_excursion_inds = (ground_truth >= threshold_low).nonzero()[:, 0]

    def compute_mismatch(coverage):
        # Plot estimated excursion set using coverage function.
        est_excursion_inds = vorobev_quantile_inds(coverage, quantile)

        # Compute mismatch.
        mismatch = np.zeros(volcano_coords.shape[0])
        mismatch[est_excursion_inds] = 1
        tmp = np.zeros(volcano_coords.shape[0])
        tmp[true_excursion_inds] = 2
        mismatch = mismatch + tmp

        excu_size = true_excursion_inds.shape[0] + 1
        p_false_pos = 100 * np.sum(mismatch == 1) / excu_size
        p_false_neg = 100 * np.sum(mismatch == 2) / excu_size
        p_correct = 100 * np.sum(mismatch == 3) / excu_size

        return (p_false_pos, p_false_neg, p_correct)

    mismatches = []
    # Compute the mismatch evolution.
    for coverage in coverages:
        mismatches.append(compute_mismatch(coverage))
    mismatches = np.array(mismatches)

    df = pd.DataFrame({'n_datapoints': n_datapoints, 'false positives': mismatches[:, 0],
            'false negatives': mismatches[:, 1], 'correct': mismatches[:,2],
            })
    return df


def plot_excu_profile_with_data(volcano_coords, data_coords_niklas, 
        ground_truth, threshold, out_file_path):
    """ Plot the x, y and z profile of the excursion set, together wiht Niklas
    data.

    """
    X_niklas = data_coords_niklas[:, 0]
    Y_niklas = data_coords_niklas[:, 1]
    Z_niklas = data_coords_niklas[:, 2]

    volcano_X = 111*1e3 * (1e-5 * (volcano_coords[:, 0] -
            np.min(volcano_coords[:, 0])))
    volcano_Y = 111*1e3 * (1e-5 * (volcano_coords[:, 1] -
            np.min(volcano_coords[:, 1])))
    volcano_Z = volcano_coords[:, 2]

    # Center compared to volano.
    X_niklas_centred = 111*1e3 * (1e-5 * (X_niklas - np.min(volcano_coords[:, 0])))
    Y_niklas_centred = 111*1e3 * (1e-5 * (Y_niklas - np.min(volcano_coords[:, 1])))

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

    f_ax1 = fig.add_subplot(gs[:, 0])
    f_ax2 = fig.add_subplot(gs[1:2, 1])
    f_ax3 = fig.add_subplot(gs[2:3, 1])

    f_ax1.scatter(volcano_X[true_excursion_inds],
            volcano_Y[true_excursion_inds], c=true_excu_densities,
            cmap=cmap,
            s=2.8,
            marker="h",
            alpha=0.2, edgecolors='none')
    f_ax1.scatter(X_niklas_centred, Y_niklas_centred, c="black", linewidth=0.2, s=0.3,
            alpha=1.0)

    f_ax1.set_aspect('equal')
    f_ax1.tick_params(axis='both', which='major', pad=0.1)


    f_ax2.scatter(volcano_X[true_excursion_inds],
            volcano_Z[true_excursion_inds], c=true_excu_densities,
            cmap=cmap,
            s=3,
            marker="h",
            alpha=0.2, edgecolors='none')
    f_ax2.scatter(X_niklas_centred, Z_niklas, c="black", linewidth=0.2, s=0.1, alpha=1.0)
    f_ax2.set_aspect(1.5)
    f_ax2.tick_params(axis='both', which='major', pad=0.1)

    f_ax3.scatter(volcano_Y[true_excursion_inds],
            volcano_Z[true_excursion_inds], c=true_excu_densities,
            cmap=cmap,
            s=3,
            marker="h",
            alpha=0.2, edgecolors='none')
    f_ax3.scatter(Y_niklas_centred, Z_niklas, c="black", linewidth=0.2, s=0.3, alpha=1.0)
    f_ax3.set_aspect(1.5)
    f_ax3.tick_params(axis='both', which='major', pad=0.1)

    plt.savefig(out_file_path, bbox_inches="tight", pad_inches=0.1, dpi=400)

def plot_excu_profile_with_visited_inds(volcano_coords, data_coords_niklas, 
        visited_coords, coast_coords,
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

    # List for coloring visited points according to progress.
    visited_colors = np.array(list(range(X_visited.shape[0])))

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

    f_ax1.set_aspect('equal')
    f_ax1.tick_params(axis='both', which='major', pad=0.1)

    f_ax1.set_xlim([min_x, max_x])
    f_ax1.set_ylim([min_y, max_y])


    f_ax2.scatter(volcano_X[true_excursion_inds],
            volcano_Z[true_excursion_inds], c=true_excu_densities,
            cmap=cmap,
            s=3,
            marker="h",
            alpha=0.2, edgecolors='none')
    f_ax2.scatter(X_niklas_centred, Z_niklas, c="black", linewidth=0.2, s=0.05, alpha=1.0)

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

    f_ax2.set_aspect(1.5)
    f_ax2.tick_params(axis='both', which='major', pad=0.1)

    f_ax2.set_xlim([min_x, max_x])
    f_ax2.set_ylim([min_z, max_z + 280.0])

    f_ax3.scatter(volcano_Y[true_excursion_inds],
            volcano_Z[true_excursion_inds], c=true_excu_densities,
            cmap=cmap,
            s=3,
            marker="h",
            alpha=0.2, edgecolors='none')
    f_ax3.scatter(Y_niklas_centred, Z_niklas, c="black", linewidth=0.2, s=0.05, alpha=1.0)

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

    f_ax3.set_aspect(1.5)
    f_ax3.tick_params(axis='both', which='major', pad=0.1)

    f_ax3.set_xlim([min_y, max_y])
    f_ax3.set_ylim([min_z, max_z + 280.0])

    plt.savefig(out_file_path, bbox_inches="tight", pad_inches=0.1, dpi=400)

def get_coast_coordinates(grid):
    """ Returns the coordinates of the coastal points of the Stromboli island.

    """
    # Manually define the coast to avoid double coastlines.
    tmp_surface_coords = grid[grid.surface_inds]
    tmp_surface_roofs = grid.cells_roof[grid.surface_inds]

    coast_2 = tmp_surface_coords[(tmp_surface_roofs > 0.0) & 
            (tmp_surface_roofs < 2.7) & (tmp_surface_coords[:, 0] > 520500.0)]
    coast_4 = tmp_surface_coords[(tmp_surface_roofs > 0.0) & 
            (tmp_surface_roofs < 12) & (tmp_surface_coords[:, 0] < 520500.0) & 
            (tmp_surface_coords[:, 0] > 519700.0) &
            (tmp_surface_coords[:, 1] > 4.294 * 1e6)]

    coast_5 = tmp_surface_coords[(tmp_surface_roofs > 0.0) & 
            (tmp_surface_roofs < 12) & (tmp_surface_coords[:, 0] < 520500.0) & 
            (tmp_surface_coords[:, 0] > 519700.0) &
            (tmp_surface_coords[:, 1] < 4.294 * 1e6)]


    coast_1 = tmp_surface_coords[(tmp_surface_roofs > 0.0) & 
            (tmp_surface_roofs < 25) & (tmp_surface_coords[:, 0] < 519700.0) & 
            (tmp_surface_coords[:, 1] > 4.2917 * 1e6)]
    coast_3 = tmp_surface_coords[(tmp_surface_roofs > 0.0) & 
            (tmp_surface_roofs < 7) & (tmp_surface_coords[:, 0] < 520000.0) & 
            (tmp_surface_coords[:, 1] <= 4.2917 * 1e6)]
    coast = np.concatenate([coast_1, coast_2, coast_3, coast_4, coast_5])
    return coast
