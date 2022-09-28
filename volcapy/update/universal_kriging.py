""" Module implementing universal kriging for inversion.

"""
import itertools
import numpy as np
import pandas as pd
import torch
import warnings
from timeit import default_timer as timer
from torch.distributions.multivariate_normal import MultivariateNormal
from volcapy.utils import _make_column_vector
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
        self.coeff_F = coeff_F.float().to(DEVICE)
        self.coeff_prior_cov = coeff_cov.float().to(DEVICE)
        self.coeff_prior_mean = coeff_mean.float().to(DEVICE)

        # Pre compute some stuff.
        self.sigma_Ft = (coeff_cov @ coeff_F.t()).to(DEVICE)

    def compute_prior_pushfwd(self, G, lambda0=None, ignore_trend=True):
        """ Given an operator G, compute the covariance pushforward (1 / sigma0^2) * K_0 G^T,
        i.e. the pushforward with respect to the prior.

        Note that the sigma0 is stripped.

        Parameters
        ----------
        G: (n_data, self.n_cells) Tensor
        lambda0: float or Tensor (1)
            Covariance kernel lengthscale parameter.
            Can be specified to override the one specified in the GP. 
            If not provided, then use the one of the GP.
        ignored_trend: bool, defaults to True
            If set to False, then include the trend part. This is 
            used when computing the posterior in the fully Bayesian case.

        Returns
        -------
        pushfwd: (self.n_cells, n_data) Tensor

        """
        # If not defined, use the one from the GP.
        if lambda0 is None: lambda0 = self.lambda0

        if not torch.is_tensor(lambda0):
            lambda0 = torch.tensor([lambda0])

        # If both devices not equal, fallback to standard device.
        if not G.device == DEVICE: G = G.to(DEVICE)

        if lambda0 is None: lambda0 = self.lambda0

        pushfwd = self.cov_module.compute_cov_pushforward(
                lambda0, G, self.cells_coords, DEVICE,
                n_chunks=self.n_chunks,
                n_flush=self.n_flush)
        if ignore_trend is True:
            # Caching.
            self._pushfwd_cache = pushfwd
            return pushfwd
        else:
            # Trend part.
            trend_pushfwd = self.coeff_F @ (self.sigma_Ft @ G.t())
            # Caching.
            self._pushfwd_cache = pushfwd + trend_pushfwd
            return pushfwd + trend_pushfwd

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
        coeff_mean: Tensor (n_cells, 1)
            Prior mean of the trend coefficients
        coeff_cov: Tensor (n_cells, n_cells)
            Prior covariance matrix of the trend coefficients.
        n_chunks: int
            Number of chunks to break the covariane matrix in.
            Increase if computations do not fit in memory.
    
        """
        if not torch.is_tensor(coeff_F): coeff_F = torch.from_numpy(coeff_F)
        # We work in single precision to allow more to fit on GPU.
        coeff_F = coeff_F.float()
        self.coeff_F = coeff_F.to(DEVICE)

        # Improper uniform prior and true Bayesian prior are handled differently.
        # Not that after having seen the first batch of data, the uniform case 
        # reduces to the Bayesian one.
        if coeff_cov == 'uniform' and coeff_mean == 'uniform':
            self.state = 'uniform'
            coeff_mean = torch.zeros(coeff_F.shape[1]).float()
            coeff_cov = torch.zeros((coeff_F.shape[1], coeff_F.shape[1])).float()
        else:
            self.state = 'bayesian'
            # Pre-process the mean and covariance parameters.
            if not torch.is_tensor(coeff_cov): coeff_F = torch.from_numpy(coeff_cov)
            if not torch.is_tensor(coeff_mean): coeff_F = torch.from_numpy(coeff_mean)
            # We work in single precision to allow more to fit on GPU.
            coeff_cov, coeff_mean = coeff_cov.float(), coeff_mean.float()
            self.coeff_prior_cov = coeff_cov.to(DEVICE)
            self.coeff_prior_mean = coeff_mean.to(DEVICE)

        self.covariance = UniversalUpdatableCovariance(cov_module, lambda0,
                sigma0, cells_coords,
                coeff_F, coeff_cov, coeff_mean,
                n_chunks)
        self.mean = UniversalUpdatableMean(coeff_F, coeff_mean,
            self.covariance)

        self.n_cells = cells_coords.shape[0]

    def predict_uniform(self, G, y, data_std):
        if not G.device == DEVICE: G = G.to(DEVICE)
        if not y.device == DEVICE: y = y.to(DEVICE)
        G = G.float()
        y = y.float()
        y = y.reshape(-1, 1)
        n = y.shape[0]

        # Compute with correlation matrix by setting sigma to 1.
        pushfwd = self.covariance.compute_prior_pushfwd(G).float()

        R = self.covariance.sigma0.float()**2 * G @ pushfwd + data_std**2 * torch.eye(G.shape[0], device=DEVICE)
        R_inv = torch.inverse(R)
        beta_hat = (
                torch.inverse(self.coeff_F.t() @ G.t() @ R_inv @ G @ self.coeff_F)
                @
                self.coeff_F.t() @ G.t() @ R_inv @ y
                )

        post_mean = (
                self.coeff_F @ beta_hat
                + self.covariance.sigma0.float()**2 * pushfwd @ R_inv @
                (y - G @ self.coeff_F @ beta_hat))
        return post_mean.cpu()

    # TODO: Finish implementing in update form.
    def update_uniform(self, G, y, data_std):
        """ Compute posterior in the case where the trend prior 
        is an improper uniform one.

        """
        if not G.device == DEVICE: G = G.to(DEVICE)
        if not y.device == DEVICE: y = y.to(DEVICE)
        G = G.float()
        y = y.float()
        y = y.reshape(-1, 1)
        n = y.shape[0]

        # Compute with correlation matrix by setting sigma to 1.
        pushfwd = self.covariance.compute_prior_pushfwd(G).float()

        # Do everything in floats.
        R = self.covariance.sigma0.float()**2 * G @ pushfwd + data_std**2 * torch.eye(G.shape[0], device=DEVICE)
        R_inv = torch.inverse(R)
        beta_hat = (
                torch.inverse(self.coeff_F.float().t() @ G.t() @ R_inv @ G @ self.coeff_F.float())
                @
                self.coeff_F.float().t() @ G.t() @ R_inv @ y
                )
        cov_beta_hat = R_inv

        # After first conditioning, the posterior on the trend parameters is Gaussian, 
        # and we are in the Bayesian kriging case.
        self.state = 'bayesian' 
        self.coeff_post_mean = beta_hat
        self.coeff_post_cov = cov_beta_hat

        self.post_mean = (
                self.coeff_F @ beta_hat
                + self.covariance.sigma0.float()**2 * pushfwd @ R_inv @ (y - G @ self.coeff_F.float() @ beta_hat))

    def compute_cv_matrix(self, G, y, data_std, sigma0=None, use_cached_pushfwd=False):
        """ Compute the cross-validation matrix K_tilde.

        Parameters
        ----------
        use_cached_pushfwd: bool, defaults to False
            If true, then use the pushforward from last computations. 
            This is to be used if lambda0 hasn't changed, so we can 
            speed up drastically.

        """
        if not G.device == DEVICE: G = G.to(DEVICE)

        # If sigma0 not specified, then use the one from the GP. 
        # Otherwise, we can use gradients for optimization.
        if sigma0 is None: sigma0 = self.sigma0

        # If only sigma0 has changed, then can used cached pushforward.
        if use_cached_pushfwd is True:
            pushfwd = self.covariance._pushfwd_cache.double()
        else:
            pushfwd = self.covariance.compute_prior_pushfwd(G).double()
        R = (
                sigma0**2 * G.double() @ pushfwd
                + data_std**2 * torch.eye(G.shape[0], device=DEVICE).double())
        K_tilde = torch.vstack([
            torch.hstack([R, G.double() @ self.coeff_F.double()]),
            torch.hstack([self.coeff_F.t().double() @ G.t().double(),
                torch.zeros((self.coeff_F.shape[1], self.coeff_F.shape[1]), device=DEVICE).double()])])
        return K_tilde

    def compute_cv_residual(self, G, y, data_std, out_inds, sigma0=None):
        """ Compute cross-validation residual at left out indices out_inds.

        Returns
        -------
        residual: Tensor (out_inds.shape[0], 1)
            Cross-validation residual when prediciting at the left out indices.
        residual_cov: Tensor (out_inds.shape[0], out_inds.shape[0])
            Cross-validation predictor covariance.

        """
        K_tilde = self.compute_cv_matrix(G, y, data_std, sigma0)
        K_tilde_inv = torch.inverse(K_tilde)
        return _compute_cv_residual(K_tilde_inv, y, out_inds)

    def _compute_cv_residual(self, K_tilde_inv, y ,out_inds):
        """ Helper function for cv residuals computation.

        """
        if not y.device == DEVICE: y = y.to(DEVICE)

        print(out_inds)
        print(K_tilde_inv[out_inds, :])
        print(K_tilde_inv[out_inds, :][:, out_inds])

        block_1 = torch.inverse(K_tilde_inv[out_inds, :][:, out_inds])
        if out_inds.shape[0] > 1:
            block_2 = (K_tilde_inv[:, :y.shape[0]] @ y.double())[out_inds, :]
        else:
            block_2 = (K_tilde_inv[:, :y.shape[0]] @ y.double())[out_inds]

        residual = block_1 @ block_2
        residual_cov = block_1
        return residual, residual_cov

    def leave_1_out_residuals(self, G, y, data_std, sigma0=None, use_cached_pushfwd=False):
        """ Return the vector of leave-one-out cross-validation erros.

        """
        K_tilde = self.compute_cv_matrix(G, y, data_std, use_cached_pushfwd, sigma0)
        K_tilde_inv = torch.inverse(K_tilde)

        residuals = torch.zeros(y.shape)
        # Loop over data points.
        for i in range(y.shape[0]):
            out_inds = np.array([i]).reshape(1,)
            residual, residual_cov = self._compute_cv_residual(K_tilde_inv, y, out_inds)
            residuals[i] = residual.item()

        return residuals

    def leave_k_out_criterion(self, k, G, y, data_std, sigma0=None, use_cached_pushfwd=False):
        """ Compute the leave k out cross validation criterion (squared errors).

        """
        K_tilde = self.compute_cv_matrix(G, y, data_std, use_cached_pushfwd, sigma0)
        K_tilde_inv = torch.inverse(K_tilde)

        criterion = 0
        # Generate all subsets with k elements.
        for i, out_inds in enumerate(itertools.combinations(list(range(y.shape[0])), k)):
            print(i)
            residual, residual_cov = self._compute_cv_residual(K_tilde_inv, y, np.array(out_inds))
            criterion += (residual**2).sum().item()
        return criterion

    def k_fold_criterion(self, folds, G, y, data_std, use_cached_pushfwd=False):
        """ Compute the k fold cross validation criterion (squared errors).

        """
        K_tilde = self.compute_cv_matrix(G, y, data_std, use_cached_pushfwd)
        K_tilde_inv = torch.inverse(K_tilde)

        criterion = 0
        # Loop over folds.
        for i, fold_inds in enumerate(folds):
            print(i)
            residual, residual_cov = self._compute_cv_residual(K_tilde_inv, y, fold_inds)
            criterion += (residual**2).sum().item()
        return criterion

    def train_cv_criterion(self, lambda0s, sigma0s, G, y, data_std,
            criterion, k=None, folds=None, out_path=None):
        """ Compute the specified cross-validation criterion on a grid of covariance 
        hyperparameters.

        Parameters
        ----------
        lambda0s: array
            List of lambda0 values.
        sigma0s: array 
            List of simga0 values.
        G: Tensor (n_data, n_cells)
            Forward operator.
        y: Tensor (n_data, 1)
            Data vector.
        data_std: float
            Observation noise standard deviation.
        out_path: string
            Where to save the pandas dataframe containing the training results.
        criterion: string
            "leave k out" or "k fold"
        k: int, defaults to None.
            If criterion is "leave k out", then specifies the number of left-out points.
        folds: List[array]
            If criterion is "k fold", then specifies the folds.
            List of integer arrays, each describing the indices 
            belonging to a given fold.

        """
        start = timer()
        # Store results in Pandas DataFrame.
        df = pd.DataFrame(columns=['lambda0', 'sigma0', 'sum squared residuals'])

        # Pick the corresponding criterion function.
        if criterion == "leave k out":
            criterion_fn = lambda: self.leave_k_out_criterion(k, G, y, data_std,
                        use_cached_pushfwd=True)
        elif criterion == "k fold":
            criterion_fn = lambda: self.k_fold_criterion(folds, G, y, data_std,
                    use_cached_pushfwd=True)

        for lambda0 in lambda0s:
            # Compute the pushforward once lambda0 has changed, 
            # so that then we can use the cached version.
            pushfwd = self.covariance.compute_prior_pushfwd(G).double()

            for sigma0 in sigma0s:
                print(sigma0)
                # Set the new parameters.
                self.lambda0 = lambda0
                self.sigma0 = sigma0

                sum_squared_residual = criterion_fn()

                df = df.append({'lambda0': lambda0, 'sigma0': sigma0,
                    'sum squared residuals': sum_squared_residual},
                    ignore_index=True)
            # Save after each lambda0.
            df.to_pickle(out_path)
        end = timer()
        print("Training done in {} minutes.".format((end-start)/60))

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
                loc=self.coeff_prior_mean.reshape(-1), covariance_matrix=self.coeff_prior_cov)
        trend_sample = distrib.rsample().float()

        return (centred_sample.reshape(-1) + self.coeff_F @ trend_sample.reshape(-1), trend_sample)

    def neg_log_likelihood(self, lambda0, sigma0, coeff_mean, coeff_cov, G, y, data_std=0.0):
        """ Compute the negative log-likelihood (up to a constant and a factor 1/2).

        Parameters
        ----------
        lambda0: Tensor
            Lengthscale parameter.
        sigma0: Tensor
            Prior standard deviation.
        coeff_mean: Tensor (n_cells, 1)
            Prior mean of the trend coefficients
        coeff_cov: Tensor (n_cells, n_cells)
            Prior covariance matrix of the trend coefficients.
        G: Tensor (n_data, n_model)
            Observation operator.
        y: Tensor (n_data)
            The data vector.
        data_std: float
            Noise variance, if not provided, then defaults to 0.

        Returns
        -------
        neg_log_likelihood: Tensor

        """
        if not G.device == DEVICE: G = G.to(DEVICE)
        if not y.device == DEVICE: y = y.to(DEVICE)

        y = y.reshape(-1, 1)

        pushfwd = self.covariance.compute_prior_pushfwd(G, lambda0).cpu()
        data_cov = (
                G @ pushfwd
                + data_std**2 * torch.eye(G.shape[0], device=DEVICE)
                + G @ self.coeff_F @ coeff_cov @ self.coeff_F.t() @ G.t())
        inv = torch.inverse(data_cov)

        prior_mean = self.coeff_F @ coeff_mean
        nll = (
                torch.logdet(data_cov)
                + 
                (y - G @ prior_mean).t() 
                @ inv 
                @ (y - G @ prior_mean))
        return nll

    def concentrated_NLL(self, lambda0, G, y, kappa_2):
        """ Concentrated negative log-likelihood for the universal kriging 
        (improper uniform prior) model.

        Parameters
        ----------
        lambda0
        G
        y
        kappa_2: float
            (squared) noise to variance ratio.

        Returns
        -------
        NLL: Tensor
        sigma_hat: Tensor
        beta_hat: Tensor

        """
        if not G.device == DEVICE: G = G.to(DEVICE)
        if not y.device == DEVICE: y = y.to(DEVICE)

        y = y.reshape(-1, 1)
        n = y.shape[0]

        # Compute with correlation matrix by setting sigma to 1.
        pushfwd = self.covariance.compute_prior_pushfwd(G, lambda0)
        R = G @ pushfwd + kappa_2 * torch.eye(G.shape[0], device=DEVICE)
        R_inv = torch.inverse(R)
        beta_hat = (
                torch.inverse(self.coeff_F.t() @ G.t() @ R_inv @ G @ self.coeff_F)
                @
                self.coeff_F.t() @ G.t() @ R_inv @ y
                )
        sigma_hat_2 = (1 / n) * (
                (y - G @ self.coeff_F @ beta_hat).t()
                @ 
                R_inv 
                @ 
                (y - G @ self.coeff_F @ beta_hat)
                )[0]
        NLL = n * torch.log(sigma_hat_2) + torch.logdet(R) + n
        return (NLL, torch.sqrt(sigma_hat_2), beta_hat)

    def train(self, lambda0s, kappa_s, data_std, G, y, out_path):
        """ Given lambda0, optimize the two remaining hyperparams via MLE.
        Here, instead of giving lambda0, we give a (stripped) covariance
        matrix. Stripped means without sigma0.

        This method is only valid for uniform trend priors.

        The user can choose between CPU and GPU.

        Parameters
        ----------
        lambda0s: Iterable
            List of prior lengthscale for which to optimize the two other hyperparams.
        kappa_s
        data_std
        G: tensor
            Measurement matrix
        y: (n_data, 1) Tensor
            Observed data. Has to be column vector.
        out_path: string
            Path for training results.

        Returns
        -------
        (sigma0, nll, train_RMSE)

        """
        start = timer()
        # Store results in Pandas DataFrame.
        df = pd.DataFrame(columns=['lambda0', 'kappa', 'sigma0', 'beta0', 'nll', 'train RMSE'])

        for lambda0 in lambda0s:
            for kappa in kappa_s:
                kappa_2 = kappa**2
                (NLL, sigma0, beta0) = self.concentrated_NLL(lambda0, G, y, kappa_2)

                # Compute prediction error on train dataset.
                pred = self.predict_uniform(G, y, data_std)
                data_pred = G @ pred
                rmse = torch.sqrt(torch.mean((data_pred.reshape(-1) - y.reshape(-1))**2))

                print(pred)
                df = df.append({'lambda0': lambda0, 'kappa': kappa, 
                    'sigma0': sigma0.cpu().numpy(), 'beta0': beta0.cpu().numpy(),
                    'nll': NLL.cpu().numpy(),
                    'train RMSE': rmse.cpu().numpy()}, ignore_index=True)
            # Save after each lambda0.
            df.to_pickle(out_path)
        end = timer()
        print("Training done in {} minutes.".format((end-start)/60))


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
