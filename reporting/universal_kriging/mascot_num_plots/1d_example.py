""" Simple 1D example for MascotNum.

"""
from volcapy.inverse.toy_fourier_1d import ToyFourier1d
import volcapy.covariance.matern52 as kernel
from volcapy.update.updatable_covariance import UpdatableGP
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plot_params = {
        'font.size': 18,
        'axes.labelsize': 'small',
        'axes.titlesize':'small',
        'legend.fontsize': 'small',
        'xtick.labelsize': 'x-small',
        'ytick.labelsize': 'x-small'
        }
plt.rcParams.update(plot_params)


output_folder = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/universal_kriging/mascot_num_plots/plots/"

n_cells_1d = 2000

my_problem = ToyFourier1d.build_problem(n_cells_1d)

m0 = 0.0
sigma0 = np.sqrt(2.0)
lambda0 = 0.05

cell_coords = torch.from_numpy(my_problem.grid.cells)
constant_updatable_gp = UpdatableGP(kernel, lambda0, torch.tensor([sigma0]), m0,
            cell_coords, n_chunks=200)

# Build some ground truth by sampling.
"""
from volcapy.covariance.sample import direct_sample
ground_truth = direct_sample(kernel, sigma0, lambda0, m0, cell_coords).numpy()
plot_basic(my_problem.grid, ground_truth)
np.save(os.path.join(output_folder, "ground_truth.npy"), ground_truth)
"""
ground_truth_notrend = torch.from_numpy(
        np.load(os.path.join(output_folder, "ground_truth.npy"))).float()

# Simple trend. 
trend = 5 * cell_coords**3
# Cutoff trend.
trend[:1000] = 0.0
trend[:500] = - 5 * cell_coords[:500] + 5 * cell_coords[500]
ground_truth = ground_truth_notrend + trend.float()

# Plot ground truth.
plot_basic(my_problem.grid, ground_truth.numpy(),
        outfile=os.path.join(output_folder, "ground_truth.png"))

# Plot ground truth with trend.
plot_basic(my_problem.grid, ground_truth.numpy(), [trend],
        outfile=os.path.join(output_folder, "ground_truth_with_trend.png"))

data_values_re = my_problem.G_re @ ground_truth.numpy()
data_values_im = my_problem.G_im @ ground_truth.numpy()

# Learn k points.
k = 4
pts_inds = np.linspace(1, 1999, k).astype(int)
G_pts = np.zeros((k, my_problem.grid.cells.shape[0]), dtype=np.float32)
G_pts[np.array(range(k)), pts_inds] = 1.0
data_std = 0.01

G_pts = torch.from_numpy(G_pts).float()
d_pts = G_pts @ ground_truth

# Observe k points.
constant_updatable_gp = UpdatableGP(kernel, lambda0, torch.tensor([sigma0]), m0,
            cell_coords, n_chunks=200)
constant_updatable_gp.update(G_pts, d_pts, data_std=0)
post_mean = constant_updatable_gp.mean_vec.cpu().numpy()
plot_basic(my_problem.grid, ground_truth.numpy(),
        additional_vals=[post_mean],
        points=[cell_coords[pts_inds], d_pts],
        outfile=os.path.join(output_folder, "posterior_points.png"))

plot_basic(my_problem.grid, ground_truth.numpy(),
        points=[cell_coords[pts_inds], d_pts],
        outfile=os.path.join(output_folder, "ground_truth_marked_pts.png"))


# Observe a few Fourier data.
constant_updatable_gp = UpdatableGP(kernel, lambda0, torch.tensor([sigma0]), m0,
            cell_coords, n_chunks=200)
G_tot = torch.from_numpy(np.vstack([
        my_problem.G_re[1, :].reshape(1, -1), 
        my_problem.G_im[1, :].reshape(1, -1),
        my_problem.G_re[5, :].reshape(1, -1), 
        my_problem.G_im[5, :].reshape(1, -1),
        my_problem.G_re[7, :].reshape(1, -1), 
        my_problem.G_im[7, :].reshape(1, -1),
        my_problem.G_re[10, :].reshape(1, -1), 
        my_problem.G_im[10, :].reshape(1, -1),
        ]))
d_tot = torch.from_numpy(np.vstack([
            data_values_re[1, :].reshape(1, -1), 
            data_values_im[1, :].reshape(1, -1),
            data_values_re[5, :].reshape(1, -1), 
            data_values_im[5, :].reshape(1, -1),
            data_values_re[7, :].reshape(1, -1), 
            data_values_im[7, :].reshape(1, -1),
            data_values_re[10, :].reshape(1, -1), 
            data_values_im[10, :].reshape(1, -1),
            ]))

constant_updatable_gp.update(G_tot, d_tot, data_std=0.01)
post_mean_2 = constant_updatable_gp.mean_vec.cpu().numpy()
plot_basic(my_problem.grid, ground_truth.numpy(),
        additional_vals=[post_mean, post_mean_2],
        points=[cell_coords[pts_inds], d_pts],
        outfile=os.path.join(output_folder, "posterior_points_and_fourier.png"))

# Now with universal kriging.
from volcapy.update.universal_kriging import UniversalUpdatableGP
coeff_F = trend.float()
updatable_gp = UniversalUpdatableGP(kernel, lambda0, sigma0,
        cell_coords,
        coeff_F, coeff_cov="uniform", coeff_mean="uniform",
        n_chunks=200)

# With point data.
updatable_gp.update_uniform(G_pts, d_pts, data_std=0.01)
post_mean_univ = updatable_gp.post_mean.cpu().numpy()
plot_basic(my_problem.grid, ground_truth.numpy(),
        additional_vals=[post_mean, post_mean_univ],
        points=[cell_coords[pts_inds], d_pts],
        outfile=os.path.join(output_folder, "posterior_points_universal.png"))

# And now with Fourier.
updatable_gp = UniversalUpdatableGP(kernel, lambda0, sigma0,
        cell_coords,
        coeff_F, coeff_cov="uniform", coeff_mean="uniform",
        n_chunks=200)
updatable_gp.update_uniform(G_tot, d_tot, data_std=0.01)
post_mean_univ_fourier = updatable_gp.post_mean.cpu().numpy()
plot_basic(my_problem.grid, ground_truth.numpy(),
        additional_vals=[post_mean, post_mean_univ_fourier],
        points=[cell_coords[pts_inds], d_pts],
        outfile=os.path.join(output_folder, "posterior_fourier_universal.png"))


def plot_basic(grid, ground_truth, additional_vals=None, points=None, outfile=None):
    # Generate regular gridding over which to plot.
    min_x = np.min(grid.cells[:])
    max_x = np.max(grid.cells[:])

    ymin = np.min(ground_truth) - 0.1
    ymax = np.max(ground_truth) + 0.1

    fig, ax = plt.subplots(1)

    ax.plot(grid.cells.reshape(-1), ground_truth, lw=1.8, label='true',
            color='black', linestyle="dashed")

    color_strings = ['steelblue', 'red']
    if additional_vals is not None:
        for i, vals in enumerate(additional_vals):
            ax.plot(grid.cells.reshape(-1), vals, lw=1.6, label='mean',
                color=color_strings[i])

    if points is not None:
        plt.scatter(points[0], points[1], color='red', facecolors='none', linewidths=1.5)

    ymin = np.min([-4, ymin])
    ymax = np.max([4, ymax])
    plt.ylim([ymin - 0.1, ymax + 0.1])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=400)
        plt.show()
        plt.close()
    else: plt.show()
