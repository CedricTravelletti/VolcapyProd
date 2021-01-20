""" Study uncertainty reduction as we add more data points.

"""
from volcapy.inverse.toy_2d_inverse_problem import ToyInverseProblem2d
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess
import volcapy.covariance.matern52 as kernel
import numpy as np
import torch


n_cells_1d = 50
n_data_1d = 1000
data_loc_offset = 1
data_loc_length = 4

my_problem = ToyInverseProblem2d.build_problem(n_cells_1d, n_data_1d,
        data_loc_offset, data_loc_length)

m0 = 1.0
sigma0 = 2.0
lambda0 = 0.2

myGP = InverseGaussianProcess(m0, sigma0, lambda0,
        torch.tensor(my_problem.grid.cells).float(), kernel)

# Build some ground truth by sampling.
ground_truth = myGP.sample_prior().detach().numpy()
np.save("ground_truth.npy", ground_truth)
my_problem.grid.plot_values(ground_truth, cmap='jet')

data_values = my_problem.G @ ground_truth
data_feed = lambda x: data_values[x]
data_std = 0.005

n_reals = 20

for i in range(1,25):
    print(i)
    subset_inds = list(range(0, my_problem.data_coords_full.shape[0], i))
    n_data = len(subset_inds)
    m_post_m, m_post_d = myGP.condition_model(
            torch.tensor(my_problem.G[subset_inds, :]),
            torch.tensor(data_values)[subset_inds], data_std)
    sigma_post = myGP.posterior_variance()
    torch.save(m_post_m, "m_post_m_{}.pt".format(n_data))
    torch.save(sigma_post, "sigma_post_{}.pt".format(n_data))

    # Save posterior covariance.
    post_cov = myGP.posterior_covariance()
    torch.save(post_cov, "post_cov_{}.pt".format(n_data))
    """
    # Sample conditional _realizations.
    for j in range(n_reals):
        print(j)
        cond_real = myGP.sample_posterior(
                torch.tensor(my_problem.G[subset_inds, :]),
                torch.tensor(data_values)[subset_inds], data_std)
        torch.save(cond_real, "cond_real_n_{}_data_{}.pt".format(j, n_data))
    """

for i in [50, 100, 200, 400, 1000]:
    print(i)
    subset_inds = list(range(0, my_problem.data_coords_full.shape[0], i))
    n_data = len(subset_inds)
    m_post_m, m_post_d = myGP.condition_model(
            torch.tensor(my_problem.G[subset_inds, :]),
            torch.tensor(data_values)[subset_inds], data_std)
    sigma_post = myGP.posterior_variance()
    torch.save(m_post_m, "m_post_m_{}.pt".format(n_data))
    torch.save(sigma_post, "sigma_post_{}.pt".format(n_data))

    # Save posterior covariance.
    post_cov = myGP.posterior_covariance()
    torch.save(post_cov, "post_cov_{}.pt".format(n_data))
    """
    # Sample conditional _realizations.
    for j in range(n_reals):
        print(j)
        cond_real = myGP.sample_posterior(
                torch.tensor(my_problem.G[subset_inds, :]),
                torch.tensor(data_values)[subset_inds], data_std)
        torch.save(cond_real, "cond_real_n_{}_data_{}.pt".format(j, n_data))
    """
