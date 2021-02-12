""" Plot the evolution of the pointwise variance as we learn more Fourier
coefficients.
The means and variances at each stage should be precomputed with the
uncertainty_reduction script.

"""
from volcapy.inverse.toy_fourier_2d import ToyFourier2d
import numpy as np
import torch
import scipy.interpolate
import matplotlib.pyplot as plt


def plot_values(grid, vals, title,
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

    gridded_data = scipy.interpolate.griddata(
            grid.cells, vals.reshape(-1), (grid_x, grid_y), method='cubic')

    plt.subplot(111)
    plt.imshow(gridded_data.T, extent=(min_x,max_x,min_y,max_y),
            origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=400)
        plt.close()
    else: plt.show()


n_cells_1d = 50

my_problem = ToyFourier2d.build_problem(n_cells_1d)

ground_truth = np.load("./ground_truth.npy").reshape(-1)

# ------------
# DATA LOADING
# ------------
fourier_cutoffs = list(range(1, my_problem.G_re.shape[0], 5))
n_situations = len(fourier_cutoffs)
count = 0
means = np.zeros(ground_truth.shape + (n_situations,))
stds = np.zeros(ground_truth.shape + (n_situations,))
subset_inds_list = [[]]


for i in fourier_cutoffs:
    means[:, count] = torch.load("./results/m_post_m_{}.pt".format(i)).reshape(-1)
    stds[:, count] = torch.load("./results/sigma_post_{}.pt".format(i))
    count += 1

for i, n_coeffs in enumerate(fourier_cutoffs):
    print(i)
    title = str(n_coeffs)

    plot_values(my_problem.grid, stds[:, i], title,
            cmap='jet', vmin=0, vmax=4,
            outfile="./results/variance_{}.png".format(i))
