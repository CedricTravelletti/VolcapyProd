""" Compare pointwise and Fourier wIVR on a universal kriging example.

"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from volcapy.inverse.toy_fourier_2d import ToyFourier2d
from volcapy.update.universal_kriging import UniversalUpdatableGP
import volcapy.covariance.matern52 as kernel
from volcapy.strategy.myopic_weighted_ivr import MyopicWIVRStrategy


n_cells_1d = 50
forward_cutoff = 400 # Only make 200 observations (Fourier and pointwise).
my_problem = ToyFourier2d.build_problem(n_cells_1d, forward_cutoff)

m0 = 0.0
sigma0 = 0.35
lambda0 = 0.23

# Built a simple trend along the x-axis.
coeff_mean = torch.tensor([[1.0]])
coeff_cov = torch.tensor([[1.0]])
coeff_F = torch.from_numpy(my_problem.grid.cells[:, 0]).reshape(-1, 1)

updatable_gp_pt = UniversalUpdatableGP(kernel, lambda0, sigma0,
            torch.tensor(my_problem.grid.cells).float(),
            coeff_F, coeff_cov, coeff_mean,
            n_chunks=200)

# ground_truth = updatable_gp_pt.sample_prior()
# np.save("ground_truth.npy", ground_truth)
ground_truth = np.load("ground_truth.npy")
my_problem.grid.plot_values(ground_truth, cmap='jet')

# Build prior realisations.
reals = []
for i in range(200):
    print(i)
    sample = updatable_gp_pt.sample_prior()
    reals.append(sample)

# Pointwise observations.
G_pt = torch.from_numpy(my_problem.G_pt).float()
y_pt = G_pt @ ground_truth
data_std_pt = 0.005

data_feed_pt = lambda x: y[x]

# Look at excursion sets.
THRESHOLD_low = 0.57
print((ground_truth > THRESHOLD_low).sum() / ground_truth.shape[0] * 100)
my_problem.grid.plot_values(ground_truth > THRESHOLD_low)

# Check conditional volume distribution.
true_volume = (ground_truth > THRESHOLD_low).int().sum()
stacked_reals = torch.stack([x.m for x in strategy.realizations], axis=0)
empirical_coverage = torch.mean((stacked_reals > THRESHOLD_low).float(), axis=0)
excu_volumes_distributions = torch.sum((stacked_reals > THRESHOLD_low).int(), axis=1)


sns.histplot(excu_volumes_distributions.reshape(-1).numpy(),
        kde=True, binwidth=50)
plt.axvline(true_volume)
plt.show()

strategy_pt = MyopicWIVRStrategy(updatable_gp_pt, my_problem.grid.cells,
            G_pt, data_feed,
            lower=THRESHOLD_low,
            prior_realizations=reals
            )

start_ind = 1250
visited_inds, observed_data = strategy_pt.run(
            start_ind_pt, n_steps=10, data_std=data_std_pt,
            max_step=0.3,
            min_step=0.0,
            )

# Plot posterior.
var_pt = updatable_gp_pt.covariance.extract_variance()
my_problem.grid.plot_values(var_pt, cmap='jet')
my_problem.grid.plot_values(
        2*(updatable_gp_pt.mean_vec > THRESHOLD_low).int()
        - (ground_truth > THRESHOLD_low).int()
        )

# Fourier observations.
data_values_re = my_problem.G_re @ ground_truth
data_values_im = my_problem.G_im @ ground_truth
data_feed_fourier = lambda x: torch.vstack([data_values_re[x], data_values_im[x]])

strategy_fourier = MyopicWIVRStrategy(updatable_gp_fourier, my_problem.fourier_inds,
            torch.from_numpy(my_problem.G), data_feed_fourier,
            lower=THRESHOLD_low, upper=None,
            prior_realizations=reals
            )

start_ind_fourier = 0
data_std_fourier = (data_values_im.std() / 400).item()

visited_inds, observed_data = strategy_fourier.run(
            start_ind_fourier, n_steps=10, data_std=data_std_fourier,
            max_step=10.0,
            min_step=0.9,
            )

# Plot posterior.
var_fourier = updatable_gp_fourier.covariance.extract_variance()
my_problem.grid.plot_values(var, cmap='jet')
my_problem.grid.plot_values(
        2*(updatable_gp_fourier.mean_vec > THRESHOLD_low).int()
        - (ground_truth > THRESHOLD_low).int()
        )
