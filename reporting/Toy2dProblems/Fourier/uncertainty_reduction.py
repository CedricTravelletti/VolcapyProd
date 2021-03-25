""" Study uncertainty reduction as we learn more coefficients of the discrete
Fourier transform (DFT).

"""
from volcapy.inverse.toy_fourier_2d import ToyFourier2d
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess
import volcapy.covariance.matern52 as kernel
import os
import numpy as np
import torch


output_folder = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/Toy2dProblems/Fourier/results_fill_big_05/"

n_cells_1d = 400
forward_cutoff = 200 # Only make 200 observations (Fourier and pointwise).
forward_cutoff = 1 # Only make 200 observations (Fourier and pointwise).

my_problem = ToyFourier2d.build_problem(n_cells_1d, forward_cutoff)
np.save(os.path.join(output_folder, "G_re.npy"), my_problem.G_re)
np.save(os.path.join(output_folder, "G_im.npy"), my_problem.G_im)

m0 = 1.0
sigma0 = 2.0
lambda0 = 0.5

myGP = InverseGaussianProcess(m0, sigma0, lambda0,
        torch.tensor(my_problem.grid.cells).float(), kernel)

# Build some ground truth by sampling.
# ground_truth = np.load(os.path.join(output_folder, "ground_truth.npy"))
ground_truth = myGP.sample_prior().detach().numpy()
np.save(os.path.join(output_folder, "ground_truth.npy"), ground_truth)
my_problem.grid.plot_values(ground_truth, cmap='jet')

data_values_re = my_problem.G_re @ ground_truth
data_values_im = my_problem.G_im @ ground_truth

# Put 0.5% noise.
data_std = 0.005 * np.std(data_values_im)


# Make pointwise observations along a space-filling sequence.
from volcapy.utils import r_sequence
fill_sequence = r_sequence(my_problem.G_re.shape[0], 2, my_problem.n_cells_1d)

# Convert to 1D index.
fill_sequence = np.array(
    [my_problem.index_to_1d(ind[0], ind[1]) for ind in fill_sequence])
# Only keep a subset.
fill_sequence = fill_sequence[:forward_cutoff]
np.save(
        os.path.join(output_folder, "fill_sequence.npy"),
        fill_sequence)

# Pointwise observations.
G_pt = np.zeros((fill_sequence.shape[0], my_problem.grid.cells.shape[0]), dtype=np.float32)
for i, ind in enumerate(fill_sequence):
    G_pt[i, ind] = 1.0
data_values_pt = G_pt @ ground_truth
data_std_pt = 0.005 * np.std(data_values_pt)

for k in range(1, fill_sequence.shape[0] + 1):
    print(k)
    # Select the first i Fourier coefficient, and stack real and imaginary
    # part.
    G_re = my_problem.G_re[:k, :]
    G_im = my_problem.G_im[:k, :]
    G_pt_sub = G_pt[:k, :]

    d_re = data_values_re[:k]
    d_im = data_values_im[:k]
    d_pt = data_values_pt[:k]

    if k == 1:
        G_pt_sub = G_pt_sub.reshape(1, -1)

    G = np.vstack([G_re, G_im])
    d = np.vstack([d_re, d_im])

    # Condition on Fourier data.
    m_post_m, m_post_d = myGP.condition_model(
            torch.tensor(G),
            torch.tensor(d), data_std)
    sigma_post = myGP.posterior_variance()
    np.save(
            os.path.join(output_folder, "m_post_fourier_{}.npy".format(k)),
            m_post_m.detach().numpy())
    np.save(
            os.path.join(output_folder, "sigma_post_fourier_{}.npy".format(k)),
            sigma_post.detach().numpy())

    # Condition on pointwise data.
    m_post_m, m_post_d = myGP.condition_model(
            torch.tensor(G_pt_sub),
            torch.tensor(d_pt), data_std_pt)
    sigma_post = myGP.posterior_variance()
    np.save(
            os.path.join(output_folder, "m_post_pt_{}.npy".format(k)),
            m_post_m.detach().numpy())
    np.save(
            os.path.join(output_folder, "sigma_post_pt_{}.npy".format(k)),
            sigma_post.detach().numpy())

    """
    # Save posterior covariance.
    post_cov = myGP.posterior_covariance()
    np.save(
            os.path.join(output_folder, "post_cov_{}.npy".format(k)),
            post_cov.detach().numpy())

    # Sample conditional _realizations.
    for j in range(n_reals):
        print(j)
        cond_real = myGP.sample_posterior(
                torch.tensor(my_problem.G[subset_inds, :]),
                torch.tensor(data_values)[subset_inds], data_std)
        torch.save(cond_real, "cond_real_n_{}_data_{}.pt".format(j, n_data))
    """
