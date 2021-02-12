""" Study uncertainty reduction as we learn more coefficients of the discrete
Fourier transform (DFT).

"""
from volcapy.inverse.toy_fourier_1d import ToyFourier1d
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess
import volcapy.covariance.matern52 as kernel
import os
import numpy as np
import torch


output_folder = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/Toy1dProblems/results/"

n_cells_1d = 1000

my_problem = ToyFourier1d.build_problem(n_cells_1d)

m0 = 1.0
sigma0 = 2.0
lambda0 = 0.5

myGP = InverseGaussianProcess(m0, sigma0, lambda0,
        torch.tensor(my_problem.grid.cells).float(), kernel)

# Build some ground truth by sampling.
ground_truth = myGP.sample_prior().detach().numpy()
np.save("./results/ground_truth.npy", ground_truth)
my_problem.grid.plot_values(ground_truth, cmap='jet')

data_values_re = my_problem.G_re @ ground_truth
data_values_im = my_problem.G_im @ ground_truth
data_std = 0.1


# Learn more Fourier coefficients each time.
for i in range(1, 100):
    myGP = InverseGaussianProcess(m0, sigma0, lambda0,
            torch.tensor(my_problem.grid.cells).float(), kernel)
    print(i)

    # Select the first i Fourier coefficient, and stack real and imaginary
    # part.
    G_re = my_problem.G_re[:i, :]
    G_im = my_problem.G_im[:i, :]

    d_re = data_values_re[:i]
    d_im = data_values_im[:i]

    G = np.vstack([G_re, G_im])
    d = np.vstack([d_re, d_im])

    m_post_m, m_post_d = myGP.condition_model(
            torch.tensor(G),
            torch.tensor(d), data_std)
    sigma_post = myGP.posterior_variance()
    np.save(
            os.path.join(output_folder, "m_post_m_{}.npy".format(i)),
            m_post_m.float().detach().numpy())
    np.save(
            os.path.join(output_folder, "sigma_post_{}.npy".format(i)),
            sigma_post.float().detach().numpy())

    # Save posterior covariance.
    post_cov = myGP.posterior_covariance()
    np.save(
            os.path.join(output_folder, "post_cov_{}.npy".format(i)),
            post_cov.float().detach().numpy())
    """
    # Sample conditional _realizations.
    for j in range(n_reals):
        print(j)
        cond_real = myGP.sample_posterior(
                torch.tensor(my_problem.G[subset_inds, :]),
                torch.tensor(data_values)[subset_inds], data_std)
        torch.save(cond_real, "cond_real_n_{}_data_{}.pt".format(j, n_data))
    """
