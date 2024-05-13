""" Generate synthetic ground truth for the universal inversion paper. 
The ground truth generated here is sampled from a GP with fault line trend, 
using hyperparameters trained on Niklas data.

"""

import sys
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from .loading import load_niklas_volcano_data
from .models import build_fault_line_model


base_folder = sys.argv[1]
output_folder = os.path.join(base_folder, "paper_universal/ground_truths")
os.makedirs(output_folder, exist_ok=True)

niklas_volcano_data = load_niklas_volcano_data(base_folder)
volcano_coords = niklas_volcano_data["volcano_coords"]

# Define GP model (Niklas trained hyperparameters).
data_std = 0.1
sigma0 = 284.66
m0 = 2139.1
lambda0 = 651.58

lambda0 = 462.384615
sigma0 = 359.717949
m0 = 605.6599

# TODO: Note that this is a hack. We KNOW the estimated trend model,
# so should just add the fluctuation around it.
# Here we sample the trend with almost no variance around the estimated parameters.
coeff_mean = torch.tensor([m0, 0.25 * m0]).reshape(-1, 1)
coeff_cov = torch.tensor([[1e-7, 0], [0, 1e-7]])

model = build_fault_line_model(
    volcano_coords, kernel, lambda0, sigma0, coeff_cov, coeff_mean
)

# Generate 50 samples.
for i in range(50):
    print("Generating sample nr {}.".format(i))
    ground_truth, true_trend_coeffs = model.sample_prior()
    noise = MultivariateNormal(
        loc=torch.zeros(n_data), covariance_matrix=data_std**2 * torch.eye(n_data)
    ).rsample()
    synth_data = G @ ground_truth + noise

    np.save(
        os.path.join(output_folder, "ground_truth_{}.npy".format(i)),
        ground_truth.cpu().numpy(),
    )
    np.save(
        os.path.join(output_folder, "data_vals_niklas_points_{}.npy".format(i)),
        synth_data.cpu().numpy(),
    )
