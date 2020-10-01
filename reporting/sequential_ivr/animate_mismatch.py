""" Plot results of run myopic excursion set reconstruction strategy using weighted IVR.

"""
import os
import torch
import numpy as np
import volcapy.covariance.exponential as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance import UpdatableGP
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud, _array_to_point_cloud
from volcapy.plotting.vtkutils import array_to_vector_cloud
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from timeit import default_timer as timer

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_23823_cells"
data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"

# Indices of the data points that are along the coast.
from volcapy.data_preparation.paths import coast_data_inds


def main():

    # Load
    F = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_full_surface.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()

    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"surface_data_coords.npy"))).float()

    ground_truth = torch.from_numpy(
            np.load(os.path.join(data_folder, "post_sample.npy")))

    # Dictionary between the original Niklas data and our discretization.
    niklas_data_inds = torch.from_numpy(
            np.load(os.path.join(data_folder, "niklas_data_inds_insurf.npy"))).long()
    niklas_coords = data_coords[niklas_data_inds].numpy()

    # PATHS ON THE VOLCANO.
    from volcapy.data_preparation.paths import paths as paths_niklas
    # Convert to indices in the full dataset.
    paths = []
    for path in paths_niklas:
        paths.append(niklas_data_inds[path].long())

    # --------------------------------
    # DEFINITION OF THE EXCURSION SET.
    # --------------------------------
    THRESHOLD_low = 700.0
    excursion_inds = (ground_truth >= THRESHOLD_low).nonzero()[:, 0]

    # Load results.
    # visited_inds = np.load("visited_inds.npy")
    # visited_inds = np.load("./results_aws/visited_inds.npy")
    visited_inds = np.load("./results_aws/bigstep/visited_inds.npy")
    # observed_data = np.load("observed_data.npy")

    def plot(visited_inds, n_tot, coverage, output_filename=None):
        # -------------------------------------------------
        # -------------------------------------------------
        # Find the Vorob'ev threshold.
        vorobev_volume = np.sum(coverage)
        def f(x):
            return (np.abs(np.sum(coverage > x) - vorobev_volume))

        from scipy.optimize import minimize
        vorobev_threshold = minimize(f, 0.5, method='nelder-mead',
                               options={'xatol': 1e-3, 'disp': True}).x
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
        # -------------------------------------------------
        # -------------------------------------------------

        # Prepare colormap.
        from matplotlib import colors
        cmap = colors.ListedColormap(["#FFFFFF00", '#0000FF04', '#FF000004', 'green'])
        boundaries = [0.0, 0.5, 1.5, 2.5, 3.1]
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

        # Generate the plot array.
        fig = plt.figure(figsize=(14, 10))
        widths = [3, 6]
        heights = [2, 2]
        gs = fig.add_gridspec(
                ncols=2, nrows=2, width_ratios=widths,
                height_ratios=heights)
    
        ax_situation= fig.add_subplot(gs[0, 0])
        ax_legend = fig.add_subplot(gs[1, 0])
        ax_anim = fig.add_subplot(gs[:, 1])

        ax_situation.axis("off")
        ax_legend.axis("off")
        ax_anim.axis("off")


        # Plot situation.
        ax_situation.scatter(data_coords[:, 0], data_coords[:, 1], c="k",
                alpha=0.1, s=1)
    
        # Plot true excursion set.
        ax_situation.scatter(
                volcano_coords[excursion_inds, 0],
                volcano_coords[excursion_inds, 1], c="r", alpha=0.07, s=2,
                label="true excursion")
        ax_situation.legend()
        ax_situation.set_xticks([], [])
        ax_situation.set_yticks([], [])

        # Plot animation.
        ax_anim.scatter(
                volcano_coords[:, 0],
                volcano_coords[:, 1], c=mismatch, cmap=cmap, norm=norm,
                s=11)
    
        # Plot visited indices.
        ax_anim.scatter(data_coords[visited_inds, 0], data_coords[visited_inds, 1],
                marker="o", c="black", s=8, alpha=1.0)
        ax_anim.scatter(data_coords[visited_inds[-1], 0],
                data_coords[visited_inds[-1], 1],
                marker="*", c="lightgreen", s=18, alpha=1.0)
        ax_anim.set_xticks([], [])
        ax_anim.set_yticks([], [])

        # Plot legend.
        excu_size = excursion_inds.shape[0] + 1
        p_false_pos = 100 * np.sum(mismatch == 1) / excu_size
        p_false_neg = 100 * np.sum(mismatch == 2) / excu_size
        p_correct = 100 * np.sum(mismatch == 3) / excu_size

        label_obs = "observations locations {} / {}".format(visited_inds.shape[0], n_tot)
        label_false_pos = "false positives {:.2f}%".format(p_false_pos)
        label_false_neg = "false negatives {:.2f}% ".format(p_false_neg)
        label_true = "correct detection {:.2f}%".format(p_correct)

        from matplotlib.lines import Line2D
        legend_elements = [
                   Line2D([0], [0], marker='o', color='w', label=label_obs,
                          markerfacecolor='black', markeredgecolor="black", markersize=15,
                          linestyle="None"),
                   Line2D([0], [0], marker='*', color='w', label="current location",
                          markerfacecolor='lightgreen',
                          markeredgecolor="lightgreen", markersize=15,
                          linestyle="None"),
                   Line2D([0], [0], marker='o', color='w',
                           label=label_false_pos,
                           markerfacecolor='b', markeredgecolor="b", markersize=15,
                           linestyle="None"),
                   Line2D([0], [0], marker='o', color='w',
                           label=label_false_neg,
                           markerfacecolor='r', markeredgecolor="r", markersize=15,
                           linestyle="None"),
                   Line2D([0], [0], marker='o', color='w',
                           label=label_true,
                           markerfacecolor='g', markeredgecolor="g", markersize=15,
                           linestyle="None")]
        ax_legend.legend(handles=legend_elements, loc='right')
    
        if output_filename is not None:
            plt.savefig(os.path.join("./ANIMATIONS_mismatch_true", output_filename), bbox_inches='tight', pad_inches=0, dpi=600)
            plt.close()
        else: plt.show()
    
        return

    for i in range(1, visited_inds.shape[0]):
        coverage = np.load("./results_aws/bigstep/coverage_{}.npy".format(i))
        plot(visited_inds[:i], visited_inds.shape[0], coverage,
                output_filename="my_anim_{}.png".format(i))

if __name__ == "__main__":
    main()
