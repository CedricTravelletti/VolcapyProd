""" Generate samples from the final conditional law for different ground
truths.

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

n_ground_truths = 15
n_samples = 40
samples = np.zeros((my_problem.grid.cells.shape[0], n_ground_truths, n_samples))
for i in range(n_ground_truths):
    print("Ground truth nr {}.".format(i))

    ground_truth = myGP.sample_prior().detach().numpy()
    np.save("./results_histo/ground_truth_{}.npy".format(i), ground_truth)

    data_values = my_problem.G @ ground_truth
    data_std = 0.005

    m_post_m, m_post_d = myGP.condition_model(
            torch.tensor(my_problem.G),
            torch.tensor(data_values), data_std)
    sigma_post = myGP.posterior_variance()
    torch.save(m_post_m, "./results_histo/m_post_m_{}.pt".format(i))
    torch.save(sigma_post, "./results_histo/sigma_post_{}.pt".format(i))
    post_cov = myGP.posterior_covariance()
    torch.save(post_cov, "./results_histo/post_cov_{}.pt".format(i))

    # Sample conditional _realizations.
    for j in range(n_samples):
        print(j)
        samples[:, i, j] = np.random.multivariate_normal(
                m_post_m.detach().numpy().reshape(-1),
                post_cov.detach().numpy())
    np.save("./results_histo/samples.npy", samples)
