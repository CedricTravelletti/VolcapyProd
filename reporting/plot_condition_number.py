import numpy as np
import torch
from meslas.means import LinearMean
from meslas.covariance.spatial_covariance_functions import Matern32
from meslas.covariance.cross_covariances import UniformMixing
from meslas.covariance.heterotopic import FactorCovariance
from meslas.geometry.grid import IrregularGrid
from meslas.random_fields import GRF
from meslas.inverse_random_fields import InverseDiscreteGRF
from meslas.sensor_plotting import DiscreteSensor
from meslas.plotting import plot_grid_values, plot_grid_probas

from torch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.utils.cholesky import psd_safe_cholesky



from volcapy.inverse.inverse_problem import InverseProblem
from volcapy.inverse.gaussian_process import GaussianProcess
import volcapy.covariance.matern52 as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud, _array_to_vtk_point_cloud


# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')

# ------------------------------------------------------
# Load Niklas Data
# ------------------------------------------------------
# Dimension of the response.
# F = np.load("/home/cedric/PHD/Dev/VolcapySIAM/reporting/F_niklas.npy")
F = np.load("/home/cedric/PHD/Dev/VolcapySIAM/reporting/F.npy")
F = torch.as_tensor(F).float()
d_obs = np.load("/home/cedric/PHD/Dev/VolcapySIAM/reporting/niklas_data_obs.npy")
grid = Grid.load("/home/cedric/PHD/Dev/VolcapySIAM/reporting/grid.pickle")
cells_coords = grid.cells

# Careful: we have to make a column vector here.
data_std = 0.1
d_obs = torch.as_tensor(d_obs)[:, None]
n_data = d_obs.shape[0]
data_cov = torch.eye(n_data)
cells_coords = torch.as_tensor(cells_coords).detach()

# HYPERPARAMETERS
sigma0_init = 221.6
m0 = 2133.8
lambda0 = 462.0


# ------------------------------------------------------
# DEFINITION OF THE MODEL
# ------------------------------------------------------
# Dimension of the response.
n_out = 1

# Spatial Covariance.
matern_cov = Matern32(lmbda=lambda0, sigma=1.0)

# Cross covariance.
cross_cov = UniformMixing(gamma0=0.2, sigmas=[sigma0_init])
covariance = FactorCovariance(
        spatial_cov=matern_cov,
        cross_cov=cross_cov,
        n_out=n_out)

# Specify mean function, here it is a linear trend that decreases with the
# horizontal coordinate.
beta0s = np.array([m0])
beta1s = np.array([
        [0.0, 0.0, 0.5]])
mean = LinearMean(beta0s, beta1s)

# Create the GRF.
myGRF = GRF(mean, covariance)

# ------------------------------------------------------
# DISCRETIZE EVERYTHING
# ------------------------------------------------------
# Create a regular square grid in 2 dims.
my_grid = IrregularGrid(cells_coords)

# Make it smaller for now.
my_grid.points = my_grid.points[:]
my_grid.n_points = my_grid.points.shape[0]

print("Working on an equilateral triangular grid with {} nodes.".format(my_grid.n_points))

# Discretize the GRF on a grid and be done with it.
# From now on we only consider locatoins on the grid.
my_discrete_grf = InverseDiscreteGRF.from_model(myGRF, my_grid)
print("Discretization done.")


n_dats = []
conds = []
# Randomly choose some data indices.
for n_data in np.linspace(1, F.shape[0], 10):
    n_data = int(n_data)
    print(n_data)

    cond_mean = []
    for i in range(8):
        # Randomly select some data points.
        data_inds = np.random.choice(np.array(range(F.shape[0])), n_data, replace=False)
        F_subset = F[data_inds, :]
        noise_std = 0.1 * torch.ones(F_subset.shape[0])
        cond_covariance = my_discrete_grf.inverse_conditional_cov(F_subset, noise_std=noise_std)
        print("Cond covariance done.")
        cond_mean.append(np.linalg.cond(cond_covariance.numpy()))
    cond_mean = np.mean(cond_mean)
    print(cond_mean)
    n_dats.append(n_data)
    conds.append(cond_mean)

np.save("n_dats.npy", n_dats)
np.save("conds.npy", conds)

