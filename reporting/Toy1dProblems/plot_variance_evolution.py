""" Plot the evolution of the pointwise variance as we add more data.
The means and variances at each stage should be precomputed with the
uncertainty_reduction script.

"""
from volcapy.inverse.toy_fourier_1d import ToyFourier1d
import numpy as np
import torch
import scipy.interpolate
import matplotlib.pyplot as plt


def plot_stds(grid, ground_truth, means, sigmas, ymin, ymax, title, outfile=None):
    """ Plot a dataset over the grid. The values should be an array
    defininf the value of the data at each grid point.

    Parameters
    ----------
    vals: (n_cells) array

    """
    # Generate regular gridding over which to plot.
    min_x = np.min(grid.cells[:])
    max_x = np.max(grid.cells[:])

    fig, ax = plt.subplots(1)
    ax.plot(grid.cells.reshape(-1), ground_truth, lw=1.5, label='true',
            color='red', linestyle="dashed")
    ax.plot(grid.cells.reshape(-1), means, lw=1, label='mean', color='blue')
    ax.fill_between(grid.cells.reshape(-1), means+sigmas, means-sigmas, facecolor='blue', alpha=0.5)
    ax.fill_between(grid.cells.reshape(-1), means+3*sigmas, means+sigmas,
            facecolor='green', alpha=0.15)
    ax.fill_between(grid.cells.reshape(-1), means-3*sigmas, means-sigmas,
            facecolor='green', alpha=0.15)

    plt.title(title)
    plt.xlim([min_x, max_x])
    plt.ylim([ymin, ymax])
    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=400)
        plt.close()
    else: plt.show()


n_cells_1d = 1000
data_loc_offset = 1
data_loc_length = 4

my_problem = ToyFourier1d.build_problem(n_cells_1d)

ground_truth = np.load("./results/ground_truth.npy").reshape(-1)
ymin = np.min(ground_truth) - 0.1
ymax = np.max(ground_truth) + 0.1

situations = np.array(list(range(1, 25)))

# ------------
# DATA LOADING
# ------------
means = np.zeros(ground_truth.shape + (situations.shape[0],))
stds = np.zeros(ground_truth.shape + (situations.shape[0],))
for i, sit in enumerate(situations):
    means[:, i] = np.load("./results/m_post_m_{}.npy".format(sit)).reshape(-1)
    stds[:, i] = np.load("./results/sigma_post_{}.npy".format(sit))

    title = str(sit)

    plot_stds(my_problem.grid, ground_truth.reshape(-1),
            means[:, i].reshape(-1), stds[:, i].reshape(-1),
            ymin, ymax, title,
            outfile="./variance_{}.png".format(i))
