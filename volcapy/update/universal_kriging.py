""" Module implementing universal kriging for inversion.

"""
import numpy as np
import torch
import warnings
from torch.distributions.multivariate_normal import MultivariateNormal
from volcapy.update.updatable_covariance import UpdatableCovariance, UpdatableMean, UpdatableGP


# General torch settings and devices.
torch.set_num_threads(8)

# Select gpu if available and fallback to cpu else.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class UniversalUpdatableCovariance(UpdatableCovariance):
    """ Universal kriging version of the UpdatableCovariance.

    """
    def __init__(self, cov_module, lambda0, sigma0, cells_coords,
            coeff_F, coeff_cov, coeff_mean,
            n_chunks=200, n_flush=50):
        """ 

        Parameters
        ----------
        cov_module: CovarianceModule
            Defines which kernel to use.
        lambda0: float
            Prior lengthscale parameter for the module.
        sigma0: float
            Prior standard deviation for the covariance.
        cells_coords: Tensor
            Coordinates of the model points.
        coeff_F: Tensor (n_cells, n_coeffs)
            Design matrix of the trend.
        coeff_cov: Tensor (n_cells, n_cells)
            Prior covariance matrix of the trend coefficients.
        coeff_mean: Tensor (n_cells, 1)
            Prior mean of the trend coefficients
        n_chunks: int
            Number of chunks to break the covariane matrix in.
            Increase if computations do not fit in memory.
        n_flush: int
            Synchronize threads and flush GPU cache every *n_flush* iterations.
            This is necessary to avoid OOM errors.
            Default is 50.

        """
        super().__init__(cov_module, lambda0, sigma0, cells_coords, n_chunks, n_flush)
        self.coeff_F = coeff_F
        self.coeff_cov = coeff_cov
        self.coeff_mean = coeff_mean

        # Pre compute some stuff.
        self.sigma_Ft = coeff_cov @ coeff_F.t()

    def compute_prior_pushfwd(self, G):
        """ Given an operator G, compute the covariance pushforward K_0 G^T,
        i.e. the pushforward with respect to the prior.

        Parameters
        ----------
        G: (n_data, self.n_cells) Tensor

        Returns
        -------
        pushfwd: (self.n_cells, n_data) Tensor

        """
        pushfwd = self.cov_module.compute_cov_pushforward(
                self.lambda0, G, self.cells_coords, DEVICE,
                n_chunks=self.n_chunks,
                n_flush=self.n_flush).cpu()
        # Trend part.
        trend_pushfwd = self.coeff_F @ (self.sigma_Ft @ G.t())
        return self.sigma0**2 * pushfwd + trend_pushfwd

    def extract_variance(self):
        """ Extracts the pointwise variance from an UpdatableCovariane module.
    
        Returns
        -------
        variances: (cov_module.n_cells) Tensor
            Variance at each point.
    
        """
        # Without the sigma0 factor.
        prior_variances = self.sigma0**2 * self.cov_module.compute_diagonal(
                self.lambda0, self.cells_coords, DEVICE,
                n_chunks=self.n_chunks, n_flush=50)

        for p, r in zip(self.pushforwards, self.inversion_ops):
            prior_variances -= torch.einsum("ij,jk,ik->i",p,r.float(),p)

        coeff_variances = torch.einsum("ij,jk,ik->i", 
                self.coeff_F, self.coeff_cov, self.coeff_F)

        return prior_variances + coeff_variances


class UniversalUpdatableGP(UpdatableGP):
    def __init__(self, cov_module, lambda0, sigma0,
            cells_coords, 
            coeff_F, coeff_cov, coeff_mean,
            n_chunks):
        """ Universal kriging version of the UpdatableGP.
    
        Parameters
        ----------
        cov_module: CovarianceModule
            Defines which kernel to use.
        lambda0: float
            Prior lengthscale parameter for the module.
        sigma0: float
            Prior standard deviation for the covariance.
        cells_coords: (n_cells, n_dims) Tensor
            Coordinates of the model points.
        coeff_F: Tensor (n_cells, n_coeffs)
            Design matrix of the trend.
        coeff_cov: Tensor (n_cells, n_cells)
            Prior covariance matrix of the trend coefficients.
        coeff_mean: Tensor (n_cells, 1)
            Prior mean of the trend coefficients
        n_chunks: int
            Number of chunks to break the covariane matrix in.
            Increase if computations do not fit in memory.
    
        """
        if not torch.is_tensor(coeff_F): coeff_F = torch.from_numpy(coeff_F)
        if not torch.is_tensor(coeff_cov): coeff_F = torch.from_numpy(coeff_cov)
        if not torch.is_tensor(coeff_mean): coeff_F = torch.from_numpy(coeff_mean)
        coeff_F, coeff_cov, coeff_mean = coeff_F.float(), coeff_cov.float(), coeff_mean.float()
        self.coeff_F = coeff_F
        self.coeff_cov = coeff_cov
        self.coeff_mean = coeff_mean

        self.covariance = UniversalUpdatableCovariance(cov_module, lambda0,
                sigma0, cells_coords,
                coeff_F, coeff_cov, coeff_mean,
                n_chunks)
        self.mean = UniversalUpdatableMean(coeff_F, coeff_mean,
            self.covariance)

        self.n_cells = cells_coords.shape[0]

    # TODO: Warning: only samples from Matern 5/2.
    def sample_prior(self):
        """ Sample from prior model.

        Returns
        -------
        sample: (self.n_cells, 1)
            Column vector of sampled values at each cells.
        trend_coeffs_sample
            Vector of sampled coefficients for the trend.

        """
        import volcapy.covariance.sample as Rinterface
        sigma0 = self.covariance.sigma0.detach().cpu().item()
        lambda0 = self.covariance.lambda0

        centred_sample = Rinterface.sample(self.covariance.cov_module, sigma0, lambda0,
                0.0, self.cells_coords)

        # Sample from the tend model.
        # We need to reshape to a row vector because of the implementation of MultivariateNormal.
        distrib = MultivariateNormal(
                loc=self.coeff_mean.reshape(-1), covariance_matrix=self.coeff_cov)
        trend_sample = distrib.rsample().float()

        return (centred_sample.reshape(-1) + self.coeff_F @ trend_sample.reshape(-1), trend_sample)


class UniversalUpdatableMean(UpdatableMean):
    """ Universal kriging version of the UpdatableMean.

    """
    def __init__(self, coeff_F, coeff_mean, cov_module):
        """ 

        Params
        ------
        coeff_F: Tensor (n_cells, n_coeffs)
            Design matrix of the trend.
        coeff_mean: Tensor (n_cells, 1)
            Prior mean of the trend coefficients
        cov_module: UpdatableCovariance

        """
        prior = coeff_F @ coeff_mean
        super().__init__(prior, cov_module)
