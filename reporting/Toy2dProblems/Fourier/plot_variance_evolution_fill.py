""" Plot the evolution of the pointwise variance as we learn more Fourier
coefficients.
The means and variances at each stage should be precomputed with the
uncertainty_reduction script.

This script compares with the situation in which we add observations along a
space filling sequence.

"""
from volcapy.inverse.toy_fourier_2d import ToyFourier2d
import os
import numpy as np
import torch
import scipy.interpolate
import matplotlib.pyplot as plt


results_folder = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/Toy2dProblems/Fourier/results_fill_big"


def plot_values(grid, vals_1, vals_2, pts, title,
        cmap=None, vmin=None, vmax=None, outfile=None):
    """ Plot a dataset over the grid. The values should be an array
    defininf the value of the data at each grid point.

    Parameters
    ----------
    vals: (n_cells) array

    """
    # Generate regular gridding over which to plot.
    min_x = np.min(grid.cells[:, 0])
    max_x = np.max(grid.cells[:, 0])
    min_y = np.min(grid.cells[:, 1])
    max_y = np.max(grid.cells[:, 1])

    grid_x, grid_y = np.mgrid[min_x:max_x:grid.n_cells_1d*(1j),
            min_y:max_y:grid.n_cells_1d*(1j)]

    gridded_data_1 = scipy.interpolate.griddata(
            grid.cells, vals_1.reshape(-1), (grid_x, grid_y), method='cubic')
    gridded_data_2 = scipy.interpolate.griddata(
            grid.cells, vals_2.reshape(-1), (grid_x, grid_y), method='cubic')

    fig, axs = plt.subplots(ncols=2)

    im0 = axs[0].imshow(gridded_data_1.T, extent=(min_x,max_x,min_y,max_y),
            origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    im1 = axs[1].imshow(gridded_data_2.T, extent=(min_x,max_x,min_y,max_y),
            origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].scatter(pts[:, 0], pts[:, 1])

    axs[0].set_xlim([min_x, max_x])
    axs[0].set_ylim([min_y, max_y])

    axs[1].set_xlim([min_x, max_x])
    axs[1].set_ylim([min_y, max_y])

    # plt.colorbar(im1, axs[1])
    plt.title(title)
    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=400)
        plt.close()
    else: plt.show()

def plot_values_separate(grid, vals_1, vals_2, pts, title,
        cmap=None, vmin=None, vmax=None, outfile1=None, outfile2=None):
    """ Plot a dataset over the grid. The values should be an array
    defininf the value of the data at each grid point.

    Parameters
    ----------
    vals: (n_cells) array

    """
    # Generate regular gridding over which to plot.
    min_x = np.min(grid.cells[:, 0])
    max_x = np.max(grid.cells[:, 0])
    min_y = np.min(grid.cells[:, 1])
    max_y = np.max(grid.cells[:, 1])

    grid_x, grid_y = np.mgrid[min_x:max_x:grid.n_cells_1d*(1j),
            min_y:max_y:grid.n_cells_1d*(1j)]

    gridded_data_1 = scipy.interpolate.griddata(
            grid.cells, vals_1.reshape(-1), (grid_x, grid_y), method='cubic')
    gridded_data_2 = scipy.interpolate.griddata(
            grid.cells, vals_2.reshape(-1), (grid_x, grid_y), method='cubic')

    fig = plt.figure()
    plt.imshow(gridded_data_1.T, extent=(min_x,max_x,min_y,max_y),
            origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.scatter(pts[:, 0], pts[:, 1])

    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])
    plt.colorbar()

    plt.savefig(outfile1, bbox_inches='tight', pad_inches=0, dpi=400)
    plt.close()

    fig = plt.figure()
    plt.imshow(gridded_data_2.T, extent=(min_x,max_x,min_y,max_y),
            origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])

    plt.savefig(outfile2, bbox_inches='tight', pad_inches=0, dpi=400)
    plt.close()

n_cells_1d = 400

my_problem = ToyFourier2d.build_problem(n_cells_1d, forward_cutoff=1)

ground_truth = np.load(os.path.join(results_folder, "ground_truth.npy")).reshape(-1)
my_problem.grid.plot_values(ground_truth, cmap='jet', vmin=-6, vmax=6)
my_problem.grid.plot_values(ground_truth, cmap='jet', vmin=-6, vmax=6,
        outfile=os.path.join(results_folder, "ground_truth.png"))

# ------------
# DATA LOADING
# ------------
n_situations = 200
means_fourier = np.zeros(ground_truth.shape + (n_situations,))
stds_fourier = np.zeros(ground_truth.shape + (n_situations,))
means_pts = np.zeros(ground_truth.shape + (n_situations,))
stds_pts = np.zeros(ground_truth.shape + (n_situations,))

subset_inds = np.load(os.path.join(results_folder, "fill_sequence.npy"))


for i in range(1, n_situations):
    print(i)
    means_fourier[:, i] = np.load(
            os.path.join(results_folder, "m_post_fourier_{}.npy".format(i))).reshape(-1)
    stds_fourier[:, i] = np.load(
            os.path.join(results_folder, "sigma_post_fourier_{}.npy".format(i)))

    means_pts[:, i] = np.load(
            os.path.join(results_folder, "m_post_pt_{}.npy".format(i))).reshape(-1)
    stds_pts[:, i] = np.load(
            os.path.join(results_folder, "sigma_post_pt_{}.npy".format(i)))

    my_problem.grid.plot_values(means_fourier[:, i], cmap='jet', vmin=-6,
            vmax=6, outfile=os.path.join(results_folder,
            "post_m_fourier_{}.png".format(i)))
    my_problem.grid.plot_values(means_pts[:, i], cmap='jet', vmin=-6,
            vmax=6, outfile=os.path.join(results_folder,
            "post_m_pts_{}.png".format(i)))

    pts = my_problem.grid.cells[subset_inds[:i], :]

    plot_values_separate(my_problem.grid, np.sqrt(stds_pts[:, i]),
            np.sqrt(stds_fourier[:, i]),
            pts,
            str(i),
            cmap='jet', vmin=0, vmax=2,
            outfile1=os.path.join(results_folder, "data_loc_{}.png".format(i)),
            outfile2=os.path.join(results_folder, "std_{}.png".format(i)))
