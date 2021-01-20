from volcapy.inverse.toy_2d_inverse_problem import ToyInverseProblem2d
import numpy as np
import torch
import scipy.interpolate
import matplotlib.pyplot as plt


def plot_values(grid, vals, title,
        x_coords, y_coords,
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

    plt.subplot(121)
    plt.imshow(gridded_data.T, extent=(min_x,max_x,min_y,max_y),
            origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.subplot(122)
    plt.scatter(x_coords, y_coords)
    plt.xlim([-2.1, 2.1])
    plt.ylim([-2.1, 2.1])
    plt.title(title)
    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=400)
        plt.close()
    else: plt.show()


n_cells_1d = 50
n_data_1d = 1000
data_loc_offset = 1
data_loc_length = 4

my_problem = ToyInverseProblem2d.build_problem(n_cells_1d, n_data_1d,
        data_loc_offset, data_loc_length)

ground_truth = np.load("./results/ground_truth.npy").reshape(-1)

n_situations = len(list(range(1,25, 4)) + [50, 100, 200, 300, 400, 600, 1000, 1500, 2000, 4000])

count = 0
means = np.zeros(ground_truth.shape + (n_situations,))
stds = np.zeros(ground_truth.shape + (n_situations,))
subset_inds_list = [[]]
for i in range(1,25, 4):
    subset_inds = list(range(0, my_problem.data_coords_full.shape[0], i))
    subset_inds_list.append(subset_inds)
    n_data = len(subset_inds)

    means[:, count] = torch.load("./results/m_post_m_{}.pt".format(n_data)).reshape(-1)
    stds[:, count] = torch.load("./results/sigma_post_{}.pt".format(n_data))
    count += 1

for i in [50, 100, 200, 300, 400, 600, 1000, 1500, 2000, 4000]:
    subset_inds = list(range(0, my_problem.data_coords_full.shape[0], i))
    subset_inds_list.append(subset_inds)
    n_data = len(subset_inds)
    means[:, count] = torch.load("./results/m_post_m_{}.pt".format(n_data)).reshape(-1)
    stds[:, count] = torch.load("./results/sigma_post_{}.pt".format(n_data))
    count += 1

for i in range(1, stds.shape[1] + 1):
    title = str(len(subset_inds_list[-i]))

    x_coords = my_problem.data_coords_full[subset_inds_list[-i], 0]
    y_coords = my_problem.data_coords_full[subset_inds_list[-i], 1]

    plot_values(my_problem.grid, stds[:, -i], title,
            x_coords, y_coords,
            cmap='jet', vmin=0, vmax=4,
            outfile="./plot_{}.png".format(i))


