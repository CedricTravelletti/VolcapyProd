""" Study uncertainty reduction as we learn more coefficients of the discrete
Fourier transform (DFT).

"""
from volcapy.inverse.toy_fourier_1d import ToyFourier1d
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess
import volcapy.covariance.matern52 as kernel
from volcapy.uq.set_estimation import vorobev_expectation_inds
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


output_folder = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/Toy1dProblems/results/"

n_cells_1d = 1000

my_problem = ToyFourier1d.build_problem(n_cells_1d)

m0 = 0.0
sigma0 = np.sqrt(2.0)
lambda0 = 0.4

myGP = InverseGaussianProcess(m0, sigma0, lambda0,
        torch.tensor(my_problem.grid.cells).float(), kernel)

# Build some ground truth by sampling.
"""
ground_truth = myGP.sample_prior().detach().numpy()
np.save("./results/ground_truth.npy", ground_truth)
"""
ground_truth = np.load("./ground_truth_cool.npy")

data_values_re = my_problem.G_re @ ground_truth
data_values_im = my_problem.G_im @ ground_truth

# Learn 3 points.
G_pts = np.zeros((3, my_problem.grid.cells.shape[0]), dtype=np.float32)
G_pts[0, 250] = 1.0
G_pts[1, 500] = 1.0
G_pts[2, 750] = 1.0
data_std = 0.01

d_pts = G_pts @ ground_truth


def plot_stds(grid, ground_truth, means, sigmas,
        pts_x, pts_y, ymin, ymax, outfile=None):
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

    ax.plot(grid.cells.reshape(-1), ground_truth, lw=1.8, label='true',
            color='black', linestyle="dashed")
    ax.plot(grid.cells.reshape(-1), means, lw=1.6, label='mean',
            color='steelblue')

    ax.fill_between(grid.cells.reshape(-1), means+2*sigmas, means-2*sigmas,
            facecolor='steelblue', alpha=0.2)

    """
    ax.fill_between(grid.cells.reshape(-1), means+3*sigmas, means+sigmas,
            facecolor='green', alpha=0.3)
    ax.fill_between(grid.cells.reshape(-1), means-3*sigmas, means-sigmas,
            facecolor='green', alpha=0.3)
    """

    plt.scatter(pts_x ,pts_y, color="red", facecolors='none', linewidths=1.5)
    # plt.yticks([1.5, 2.0, 2.5, 3.0, 3.5])
    # plt.yticks([])

    plt.xlim([min_x, max_x])
    ymin = np.min([-4, ymin])
    ymax = np.max([4, ymax])
    plt.ylim([ymin - 0.1, ymax + 0.1])

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=400)
        plt.close()
    else: plt.show()

def plot_excu_plugin(grid, ground_truth, means, sigmas,
        pts_x, pts_y, ymin, ymax, excu_inds_plugin):
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
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    # Plot true excursion.
    excu_inds_true = np.argwhere(ground_truth >= THRESHOLD)[:, 0]
    ax.scatter(grid.cells.reshape(-1)[excu_inds_true],
            ground_truth[excu_inds_true], label='true',
            s=20,
            color='black', alpha=0.05)

    # Plot where mean is in excursion.
    ax.scatter(grid.cells.reshape(-1)[excu_inds_plugin],
            means[excu_inds_plugin], label='true',
            s=20,
            color='blue', alpha=0.02)

    ax.plot(grid.cells.reshape(-1), ground_truth, lw=1.8, label='true',
            color='black', linestyle="dashed")
    ax.plot(grid.cells.reshape(-1), means, lw=1.6, label='mean',
            color='steelblue')

    ax.fill_between(grid.cells.reshape(-1), means+2*sigmas, means-2*sigmas,
            facecolor='steelblue', alpha=0.2)

    # Horizontal line at excursion level.
    ax.plot(grid.cells.reshape(-1),
            np.repeat(THRESHOLD, grid.cells.shape[0]), lw=1.0, label='true',
            color='red', linestyle="dashed")

    # Plost observation points.
    ax.scatter(pts_x ,pts_y, color="red", facecolors='none', linewidths=1.5)

    ymin = np.min([-4, ymin])
    ymax = np.max([4, ymax])
    ax.set_ylim(ymin - 0.1, ymax + 0.1)

    plt.xlim([min_x, max_x])

    ax2.set_ylabel('A', color='white', alpha=0.0)  # we already handled the x-label with
    ax2.set_yticks([0.0, 0.2])
    ax2.tick_params(axis='y', labelcolor='white', color='white')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.savefig("./excu_plugin.png", bbox_inches='tight', pad_inches=0, dpi=400)
    plt.close()

def plot_excu_coverage(grid, ground_truth, means, sigmas,
        pts_x, pts_y, ymin, ymax, excu_inds_plugin, coverage):
    # Generate regular gridding over which to plot.
    min_x = np.min(grid.cells[:])
    max_x = np.max(grid.cells[:])

    fig, ax = plt.subplots(1)
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    ax.plot(grid.cells.reshape(-1), ground_truth, lw=1.8, label='true',
            color='black', linestyle="dashed")
    ax.plot(grid.cells.reshape(-1), means, lw=1.6, label='mean',
            color='steelblue')

    ax.fill_between(grid.cells.reshape(-1), means+2*sigmas, means-2*sigmas,
            facecolor='steelblue', alpha=0.2)

    # Horizontal line at excursion level.
    ax.plot(grid.cells.reshape(-1),
            np.repeat(THRESHOLD, grid.cells.shape[0]), lw=1.0, label='true',
            color='red', linestyle="dashed")

    # Plost observation points.
    ax.scatter(pts_x ,pts_y, color="red", facecolors='none', linewidths=1.5)

    ymin = np.min([-4, ymin])
    ymax = np.max([4, ymax])
    ax.set_ylim(ymin - 0.1, ymax + 0.1)

    # Plot the excursion probability.
    color = 'firebrick'
    ax2.set_ylabel('Excursion Probability', color=color)  # we already handled the x-label with
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax2.set_ylim(0, 1)
    ax2.plot(grid.cells.reshape(-1), coverage, lw=1.8, label='true',
            color=color, linestyle="solid")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # --------------------------------

    plt.xlim([min_x, max_x])

    plt.savefig("./excu_coverage.png", bbox_inches='tight', pad_inches=0, dpi=400)
    plt.close()


def plot_excu_vorb_quantile(grid, ground_truth, means, sigmas,
        pts_x, pts_y, ymin, ymax, excu_inds_plugin, coverage):
    # Generate regular gridding over which to plot.
    min_x = np.min(grid.cells[:])
    max_x = np.max(grid.cells[:])

    fig, ax = plt.subplots(1)
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    ax.plot(grid.cells.reshape(-1), ground_truth, lw=1.8, label='true',
            color='black', linestyle="dashed")

    # Plost observation points.
    ax.scatter(pts_x ,pts_y, color="red", facecolors='none', linewidths=1.5)

    ymin = np.min([-4, ymin])
    ymax = np.max([4, ymax])
    ax.set_ylim(ymin - 0.1, ymax + 0.1)

    # Plot Vorobev quantiles.
    excu_inds_quant_05 = np.argwhere(coverage >= 0.5)[0, :]
    excu_inds_quant_075 = np.argwhere(coverage >= 0.75)[0, :]
    ax.scatter(grid.cells.reshape(-1)[excu_inds_quant_05],
            np.repeat(THRESHOLD, grid.cells.shape[0])[excu_inds_quant_05], label='true',
            s=60,
            color='steelblue', alpha=0.9)
    ax.scatter(grid.cells.reshape(-1)[excu_inds_quant_075],
            np.repeat(THRESHOLD, grid.cells.shape[0])[excu_inds_quant_075], label='true',
            s=30,
            color='lightskyblue', alpha=0.9)

    # Horizontal line at excursion level.
    ax.plot(grid.cells.reshape(-1),
            np.repeat(THRESHOLD, grid.cells.shape[0]), lw=1.0, label='true',
            color='red', linestyle="dashed")


    # Plot the excursion probability.
    color = 'firebrick'
    ax2.set_ylabel('Excursion Probability', color=color)  # we already handled the x-label with
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax2.set_ylim(0, 1)
    ax2.plot(grid.cells.reshape(-1), coverage, lw=1.8, label='true',
            color=color, linestyle="solid")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # --------------------------------

    plt.xlim([min_x, max_x])

    plt.savefig("./excu_vorb_quantile.png", bbox_inches='tight', pad_inches=0, dpi=400)
    plt.close()

def plot_excu_vorb_exp(grid, ground_truth, means, sigmas,
        pts_x, pts_y, ymin, ymax, excu_inds_plugin, coverage):
    # Generate regular gridding over which to plot.
    min_x = np.min(grid.cells[:])
    max_x = np.max(grid.cells[:])

    fig, ax = plt.subplots(1)
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    ax.plot(grid.cells.reshape(-1), ground_truth, lw=1.8, label='true',
            color='black', linestyle="dashed")

    # Plost observation points.
    ax.scatter(pts_x ,pts_y, color="red", facecolors='none', linewidths=1.5)

    ymin = np.min([-4, ymin])
    ymax = np.max([4, ymax])
    ax.set_ylim(ymin - 0.1, ymax + 0.1)

    # Plot Vorobev expectation.
    excu_inds_vorb_exp = vorobev_expectation_inds(coverage)

    ax.scatter(grid.cells.reshape(-1)[excu_inds_vorb_exp],
            np.repeat(THRESHOLD, grid.cells.shape[0])[excu_inds_vorb_exp], label='true',
            s=50,
            color='steelblue', alpha=0.9)

    # Horizontal line at excursion level.
    ax.plot(grid.cells.reshape(-1),
            np.repeat(THRESHOLD, grid.cells.shape[0]), lw=1.0, label='true',
            color='red', linestyle="dashed")


    # Plot the excursion probability.
    color = 'firebrick'
    ax2.set_ylabel('Excursion Probability', color=color)  # we already handled the x-label with
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax2.set_ylim(0, 1)
    ax2.plot(grid.cells.reshape(-1), coverage, lw=1.8, label='true',
            color=color, linestyle="solid")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # --------------------------------

    plt.xlim([min_x, max_x])

    plt.savefig("./excu_vorb_exp.png", bbox_inches='tight', pad_inches=0, dpi=400)
    plt.close()


ymin = np.min(ground_truth) - 0.1
ymax = np.max(ground_truth) + 0.1

# PRIOR.
pts_x = np.array([my_problem.grid.cells[250],
        my_problem.grid.cells[500], my_problem.grid.cells[750]])
pts_y = d_pts

m_prior = myGP.m0 * torch.ones(
        (myGP.n_model, 1), dtype=torch.float32)
sigma_prior = myGP.sigma0**2 * torch.ones(
        (myGP.n_model, 1), dtype=torch.float32)

plot_stds(my_problem.grid, ground_truth.reshape(-1),
        m_prior.numpy().reshape(-1), sigma_prior.numpy().reshape(-1),
        pts_x, pts_y,
        ymin, ymax,
            outfile="./variance_prior.png")

# 0) Only point data.
i = 1

m_post_m, m_post_d = myGP.condition_model(
            torch.tensor(G_pts),
            torch.tensor(d_pts), data_std)
sigma_post = np.sqrt(myGP.posterior_variance())


plot_stds(my_problem.grid, ground_truth.reshape(-1),
        m_post_m.numpy().reshape(-1), sigma_post.numpy().reshape(-1),
        pts_x, pts_y,
        ymin, ymax,
            outfile="./variance_{}.png".format(i))

# -------------------
# Excursion Set Stuff
# -------------------
THRESHOLD = 1.0
excu_inds_plugin, _ = np.argwhere(m_post_m >= THRESHOLD)
coverage = myGP.coverage(THRESHOLD, upper=None)

plot_excu_plugin(my_problem.grid, ground_truth.reshape(-1),
        m_post_m.numpy().reshape(-1), sigma_post.numpy().reshape(-1),
        pts_x, pts_y,
        ymin, ymax,
        excu_inds_plugin)
plot_excu_coverage(my_problem.grid, ground_truth.reshape(-1),
        m_post_m.numpy().reshape(-1), sigma_post.numpy().reshape(-1),
        pts_x, pts_y,
        ymin, ymax,
        excu_inds_plugin, coverage)
plot_excu_vorb_quantile(my_problem.grid, ground_truth.reshape(-1),
        m_post_m.numpy().reshape(-1), sigma_post.numpy().reshape(-1),
        pts_x, pts_y,
        ymin, ymax,
        excu_inds_plugin, coverage)
plot_excu_vorb_exp(my_problem.grid, ground_truth.reshape(-1),
        m_post_m.numpy().reshape(-1), sigma_post.numpy().reshape(-1),
        pts_x, pts_y,
        ymin, ymax,
        excu_inds_plugin, coverage)


# 1) Add integral data.
G_tot = np.vstack([G_pts, my_problem.G_re[0, :].reshape(1, -1)])
d_tot = np.vstack([d_pts,
            data_values_re[0, :].reshape(1, -1)])

i = 2
m_post_m, m_post_d = myGP.condition_model(
            torch.tensor(G_tot),
            torch.tensor(d_tot), data_std)
sigma_post = np.sqrt(myGP.posterior_variance())
plot_stds(my_problem.grid, ground_truth.reshape(-1),
        m_post_m.numpy().reshape(-1), sigma_post.numpy().reshape(-1),
        pts_x, pts_y,
        ymin, ymax,
            outfile="./variance_{}.png".format(i))

# 2) Add Fourier data.
G_tot = np.vstack([G_pts,
        my_problem.G_re[0, :].reshape(1, -1), 
        my_problem.G_re[1, :].reshape(1, -1), 
        my_problem.G_im[1, :].reshape(1, -1)])
d_tot = np.vstack([d_pts,
            data_values_re[0, :].reshape(1, -1), 
            data_values_re[1, :].reshape(1, -1), 
            data_values_im[1, :].reshape(1, -1)])

i = 3
m_post_m, m_post_d = myGP.condition_model(
            torch.tensor(G_tot),
            torch.tensor(d_tot), data_std)
sigma_post = np.sqrt(myGP.posterior_variance())
plot_stds(my_problem.grid, ground_truth.reshape(-1),
        m_post_m.numpy().reshape(-1), sigma_post.numpy().reshape(-1),
        pts_x, pts_y,
        ymin, ymax,
            outfile="./variance_{}.png".format(i))

# 3) Add more Fourier data.
G_tot = np.vstack([G_pts,
        my_problem.G_re[0, :].reshape(1, -1), 
        my_problem.G_re[1:3, :], 
        my_problem.G_im[1:3, :]])
d_tot = np.vstack([d_pts,
            data_values_re[0, :].reshape(1, -1), 
            data_values_re[1:3, :], 
            data_values_im[1:3, :]])

i = 4
m_post_m, m_post_d = myGP.condition_model(
            torch.tensor(G_tot),
            torch.tensor(d_tot), data_std)
sigma_post = np.sqrt(myGP.posterior_variance())
plot_stds(my_problem.grid, ground_truth.reshape(-1),
        m_post_m.numpy().reshape(-1), sigma_post.numpy().reshape(-1),
        pts_x, pts_y,
        ymin, ymax,
            outfile="./variance_{}.png".format(i))

# 4) Add more Fourier data.
G_tot = np.vstack([G_pts,
        my_problem.G_re[0, :].reshape(1, -1), 
        my_problem.G_re[1:10, :], 
        my_problem.G_im[1:10, :]])
d_tot = np.vstack([d_pts,
            data_values_re[0, :].reshape(1, -1), 
            data_values_re[1:10, :], 
            data_values_im[1:10, :]])

i = 5
m_post_m, m_post_d = myGP.condition_model(
            torch.tensor(G_tot),
            torch.tensor(d_tot), data_std)
sigma_post = np.sqrt(myGP.posterior_variance())
plot_stds(my_problem.grid, ground_truth.reshape(-1),
        m_post_m.numpy().reshape(-1), sigma_post.numpy().reshape(-1),
        pts_x, pts_y,
        ymin, ymax,
            outfile="./variance_{}.png".format(i))

# Now to the excursion set part.

