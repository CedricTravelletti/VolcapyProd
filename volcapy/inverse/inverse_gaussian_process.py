"""
HEAVY REFACTOR OF THE GP CLASS.

This class implements gaussian process regression/conditioning for inverse
problems, i.e. when conditioning on a linear operator of the field.
The situation is we have a gaussian process on some space and a linear
operator, denoted :math:`F`, acting on the process and mapping it to another space.
This linear operator induces a gaussian process on the target space.
We will use the term *model* when we refer to the original gaussian process and
*data* when we refer to the induced gaussian process.
We discretize model-space and data-space, so that F becomes a matrix.

Notation and Implementation
---------------------------
We denote by :math:`K` the covariance matrix of the original field. We assume a
constant prior mean vector :math:`\mu_0 = m_0 1_m`, where :math:`1_m` stands for the identity
vector in model space. Hence we only have one scalar parameter m0 for the prior mean.

The induced gaussian process then has covariance matrix :math:`K_d = F K F^t` and
prior mean vector :math:`m_0 F ~ 1_m`.

Regression/Conditioning
-----------------------
We observe the value of the induced field at some points in data-space. We then
condition our GP on those observations and obtain posterior mean vector /
covariance matrix.

Covariance Matrices
-------------------
Most kernels have a variance parameter :math:`\sigma_0^2` that just appears as a
multiplicative constant. To make its optimization easier, we strip it of the
covariance matrix, store it as a model parameter (i.e. as an attribute of the
class) and include it manually in the experessions where it shows up.

This means that when one sees a covariance matrix in the code, it generally
doesn't include the :math:`\sigma_0` factor, which has to be included by hand.

Covariance Pushforward
----------------------
The model covariance matrix :math:`K` is to big to be stored, but during conditioning
it only shows up in the form :math:`K F^t`. It is thus sufficient to compute this
product once and for all. We call it the *covariance pushforward*.

Noise
-----
Independent homoscedactic noise on the data is assumed. It is specified by
*data_std_orig*. If the matrices are ill-conditioned, the the noise will have
to be increased, hence the current value of the noise will be stored in
*noise_std*.

Important Implementation Detail
-------------------------------
Products of vectors with the inversion operator R^{-1} * x should be computed
using inv_op_vector_mult(x).

Conditioning
------------
Conditioning always compute the inversion operator, and m0 via concentration.
Hence, every call to conditioning (condition_data or condition_model), will
update the following attributes:
    - inv_op_L
    - inversion_operator
    - m0
    - mu0_d
    - prior_misfit
    - weights
    - mu_post_d (TODO: maybe shouldn't be an attribute of the GP.)

"""
import numpy as np
import torch
import pandas as pd
from timeit import default_timer as timer
from volcapy.gaussian_cdf import gaussian_cdf
from volcapy.utils import _make_column_vector


class InverseGaussianProcess(torch.nn.Module):
    """

    Attributes
    ----------
    m0: float
        Current value of the prior mean of the process.
        Will get optimized. We store it so we can use it as starting value
        for optimization of similar lambda0s.
    sigma0: float
        Current value of the prior standard deviation of the process.
        Will get optimized. We store it so we can use it as starting value
        for optimization of similar lambda0s.
    lambda0: float
        Current value of the prior lengthscale of the process.
        Will get optimized. We store it so we can use it as starting value
        for optimization of similar lambda0s.
    cov_module: CovarianceModule
        Covariance kernel to use
    output_device: toch.device
        GPU to use to store the main results. Defaults to GPU0.
    logger
        An instance of logging.Logger, used to output training progression.

    """
    def __init__(self, m0, sigma0, lambda0,
            cells_coords, cov_module,
            n_chunks=200, n_flush=50,
            device=None,
            logger=None):
        """

        Parameters
        ----------
        m0: float
            Prior mean.
        sigam0: float
            Prior standard deviation.
        lambda0: float
            Prior lengthscale.
        cells_coords: tensor
            Models cells, fixed once and for all.
        cov_module: CovarianceModule
            Kernel to use.
        n_chunks: int
            Number of chunks to split the matrix into.
            Default is 200. Increase if get OOM errors.
        n_flush: int
            Synchronize threads and flush GPU cache every *n_flush* iterations.
            This is necessary to avoid OOM errors.
            Default is 50.
        logger: Logger

        """
        super(InverseGaussianProcess, self).__init__()
        
        # Get GPUS.
        if device is None:
            # Check if GPU available and otherwise go for CPU.
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.m0 = torch.tensor(m0, requires_grad=False)
        self.sigma0 = torch.tensor(sigma0, requires_grad=True)
        self.sigma0 = torch.nn.Parameter(self.sigma0)
        self.lambda0 = lambda0

        self.cells_coords = cells_coords
        self.n_model = cells_coords.shape[0]

        self.kernel = cov_module

        # The heavy part is to compute the pushforward. It is needed by all
        # other operations. Hence, if it has already been computed by an
        # operation, just leave it as is.
        self.is_precomputed_pushfwd = False

        self.n_chunks = n_chunks
        self.n_flush = n_flush

        # If no logger, create one.
        if logger is None:
            import logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
        self.logger = logger

    def compute_pushfwd(self, G):
        """ Given a measurement operator, compute the associated covariance
        pushforward K G^T (without the sigma0^2 prefactor).

        """
        # Compute the compute_covariance_pushforward and data-side covariance matrix
        self.pushfwd = self.kernel.compute_cov_pushforward(
                self.lambda0, G, self.cells_coords,
                device=self.device,
                n_chunks=self.n_chunks, n_flush=self.n_flush)
        self.K_d = G.to(self.device).double() @ self.pushfwd.double()

    def condition_data(self, G, y, data_std, concentrate=False,
            is_precomp_pushfwd=False, device=None):
        """ Given a bunch of measurement, condition model on the data side.
        I.e. only compute the conditional law of the data vector G Z, not of Z
        itself.

        Parameters
        ----------
        G: tensor
            Measurement matrix
        y: (n_data, 1) Tensor
            Observed data. Has to be column vector.
        data_std: flot
            Data noise standard deviation.
        concentrate: bool
            If true, then will compute m0 by MLE via concentration of the
            log-likelihood instead of using the current value of the
            hyperparameter.
        is_precomp_pushfwd: bool
            Set to True if the covariance pushforward has already been computed
            by a previous operation. Can be used to speedup calculations if
            previous calculations have already computed the pushforward.
        device: torch.device
            Device on which to perform the training. Should be the same as the
            one the inputs are located on.
            If None, defaults to gpu0.
        hypothetical: bool, default=False
            If set to true, then the internals of the GP (pushfwd,
            inversion_op) are not updated.
            Use when considering hypothetical data.

        Returns
        -------
        mu_post_d: tensor
            Posterior mean data vector
        nll: tensor
            Negative log likelihood.
        float
            When noise has to be increased to make matrices invertible, this
            gives the new value of the noise standard deviation.

        """
        if device is None:
            device = self.device
        if not is_precomp_pushfwd:
            self.compute_pushfwd(G)

        y = _make_column_vector(y).to(device)

        # Get inversion operator.
        self.inversion_operator, data_std = self.get_inversion_op(self.K_d, data_std)
            
        if concentrate:
            # Determine m0 (on the model side) from sigma0 by concentration of the Ll.
            self.m0 = self.concentrate_m0(G, y)

        # Prior mean (vector) on the data side.
        mu0_d_stripped = (G.to(device) @ torch.ones((self.n_model, 1),
                dtype=torch.float32, device=device))
        mu0_d = self.m0 * mu0_d_stripped
        prior_misfit = y.double() - mu0_d.double()

        self.weights = self.inversion_operator @ prior_misfit

        m_post_d = mu0_d + torch.mm(self.sigma0**2 * self.K_d, self.weights)

        nll = self.neg_log_likelihood(y, G, self.m0)

        return m_post_d.float(), nll.float(), data_std

    def neg_log_likelihood(self, y, G, m0, device=None):
        """ Computes the negative log-likelihood of the current state of the
        model.
        Note that this function should be called AFTER having run a
        conditioning, since it depends on the inversion operator computed
        there.

        Params
        ------
        y: (n_data, 1) Tensor
            Observed data. Has to be column vector.
        G: tensor
            Measurement matrix
        m0: float
        device: torch.device
            Device on which to perform the training. Should be the same as the
            one the inputs are located on.
            If None, defaults to gpu0.
        Returns
        -------
        float

        """
        if device is None:
            device = self.device

        y = _make_column_vector(y)

        # WARNING!!! determinant is not linear! Taking constants outside adds
        # power to them.
        log_det = torch.logdet(self.R).double()

        mu0_d_stripped = torch.mm(G.double().to(device), torch.ones((self.n_model, 1),
                dtype=torch.float64, device=device))
        mu0_d = m0 * mu0_d_stripped
        prior_misfit = y.double() - mu0_d.double()

        nll = log_det + torch.mm(prior_misfit.t(), self.weights)

        return nll

    def concentrate_m0(self, G, y, device=None):
        """ Compute m0 (prior mean parameter) by MLE via concentration.

        Note that the inversion operator should have been updated first.

        """
        if device is None:
            # Check if GPU available and otherwise go for CPU.
            device = self.device

        y = _make_column_vector(y).double().to(device)

        # Prior mean (vector) on the data side.
        mu0_d_stripped = torch.mm(G.double().to(device), torch.ones((self.n_model, 1),
                dtype=torch.float64, device=device))
        # Compute R^(-1) * G * I_m.
        tmp = self.inversion_operator @ mu0_d_stripped
        conc_m0 = (y.t() @ tmp) / (mu0_d_stripped.t() @ tmp)

        return conc_m0.float()

    def condition_model(self, G, y, data_std, concentrate=False,
            is_precomp_pushfwd=False, device=None, hypothetical=False):
        """ Given a bunch of measurement, condition model on the data side.
        I.e. only compute the conditional law of the data vector G Z, not of Z
        itself.

        Parameters
        ----------
        G: tensor
            Measurement matrix
        y: (n_data, 1) Tensor
            Observed data. Has to be column vector.
        data_std: flot
            Data noise standard deviation.
        concentrate: bool
            If true, then will compute m0 by MLE via concentration of the
            log-likelihood instead of using the current value of the
            hyperparameter.
        is_precomp_pushfwd: bool
            Set to True if the covariance pushforward has already been computed
            by a previous operation. Can be used to speedup calculations if
            previous calculations have already computed the pushforward.
        device: torch.device
            Device on which to perform the training. Should be the same as the
            one the inputs are located on.
            If None, defaults to gpu0.
        hypothetical: bool, default=False
            If set to true, then the internals of the GP (pushfwd,
            inversion_op) are not updated.
            Use when considering hypothetical data.

        Returns
        -------
        mu_post_m
            Posterior mean model vector
        mu_post_d
            Posterior mean data vector

        """
        if device is None:
            device = self.device

        y = _make_column_vector(y).to(device)

        # Conditioning model is just conditioning on data and then computing
        # posterior mean and (co-)variance on model side.
        m_post_d, nll, data_std = self.condition_data(G, y, data_std, concentrate=concentrate,
                is_precomp_pushfwd=is_precomp_pushfwd)

        # Posterior model mean.
        # Can re-use the m0 and weights computed by condition_data.
        if concentrate:
            m0 = self.concentrate_m0(G, y)
        else: m0 = self.m0

        m_post_m = (
                m0 * torch.ones((self.n_model, 1), dtype=torch.float64,
                        device=device)
                + (self.sigma0.double()**2 * self.pushfwd.double() @ self.weights))

        # Save in case.
        self.m_post_m = m_post_m.float()

        return self.m_post_m, m_post_d.float()

    def train_fixed_lambda(self, lambda0, G, y, data_std,
            device=None,
            n_epochs=5000, lr=0.1,
            n_chunks=200, n_flush=50):
        """ Given lambda0, optimize the two remaining hyperparams via MLE.
        Here, instead of giving lambda0, we give a (stripped) covariance
        matrix. Stripped means without sigma0.

        The user can choose between CPU and GPU.

        Parameters
        ----------
        lambda0: float
            Prior lengthscale for which to optimize the two other hyperparams.
        G: tensor
            Measurement matrix
        y: (n_data, 1) Tensor
            Observed data. Has to be column vector.
        data_std: flot
            Data noise standard deviation.
        device: torch.device
            Device on which to perform the training. Should be the same as the
            one the inputs are located on.
            If None, defaults to gpu0.
        n_epochs: int
            Number of training epochs.
        lr: float
            Learning rate.

        Returns
        -------
        (sigma0, nll, train_RMSE)

        """
        if device is None:
            device = self.device

        y = _make_column_vector(y).to(device)

        # Compute the pushforward once and for all, since it only depends on
        # lambda0 and G.
        self.lambda0 = lambda0
        self.compute_pushfwd(G)

        # Make column vector.

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(n_epochs + 1):
            # Forward pass: Compute predicted y by passing
            # x to the model
            m_post_d, nll, data_std = self.condition_data(G, y, data_std, concentrate=True,
                is_precomp_pushfwd=True, device=device)

            # Zero gradients, perform a backward pass,
            # and update the weights.
            optimizer.zero_grad()
            nll.backward(retain_graph=True)
            optimizer.step()

            # Periodically print informations.
            if epoch % 100 == 0:
                # Compute train error.
                train_RMSE = torch.sqrt(torch.mean(
                        (y - m_post_d)**2))
                self.logger.info("Epoch: {}/{}".format(epoch, n_epochs))
                self.logger.info("lambda0: {}".format(lambda0))
                self.logger.info("sigma0: {}".format(self.sigma0.item()))
                self.logger.info("m0: {}".format(self.m0.item()))
                self.logger.info("Log-likelihood: {}".format(nll.item()))
                self.logger.info("RMSE train error: {}".format(train_RMSE.item()))

        return (self.sigma0.item(), self.m0.item(), nll.item(), train_RMSE.item())

    def train(self, lambda0s, G, y, data_std,
            out_path, device=None,
            n_epochs=5000, lr=0.1,
            n_chunks=200, n_flush=50):
        """ Given lambda0, optimize the two remaining hyperparams via MLE.
        Here, instead of giving lambda0, we give a (stripped) covariance
        matrix. Stripped means without sigma0.

        The user can choose between CPU and GPU.

        Parameters
        ----------
        lambda0s: Iterable
            List of prior lengthscale for which to optimize the two other hyperparams.
        G: tensor
            Measurement matrix
        y: (n_data, 1) Tensor
            Observed data. Has to be column vector.
        data_std: flot
            Data noise standard deviation.
        out_path: string
            Path for training results.
        device: torch.device
            Device on which to perform the training. Should be the same as the
            one the inputs are located on.
            If None, defaults to gpu0.
        n_epochs: int
            Number of training epochs.
        lr: float
            Learning rate.

        Returns
        -------
        (sigma0, nll, train_RMSE)

        """
        start = timer()
        if device is None:
            device = self.device

        y = _make_column_vector(y)

        # Store results in Pandas DataFrame.
        df = pd.DataFrame(columns=['lambda0', 'sigma0', 'm0', 'nll', 'train_RMSE'])

        for lambda0 in lambda0s:
            (sigma0, m0, nll, train_RMSE) = self.train_fixed_lambda(
                    lambda0, G, y, data_std,
                    device=device, n_epochs=n_epochs, lr=lr,
                    n_chunks=n_chunks, n_flush=n_flush)
            df = df.append({'lambda0': lambda0, 'sigma0': sigma0,
                    'm0': m0,
                    'nll': nll, 'train_RMSE': train_RMSE}, ignore_index=True)
            # Save after each lambda0.
            df.to_pickle(out_path)

        end = timer()
        print("Training done in {} minutes.".format((end-start)/60))
        return df

    def get_inversion_op(self, K_d, data_std):
        """ Compute the Cholesky decomposition of the inverse of the inversion operator.
        Increases noise level if necessary to make matrix invertible.

        Note that this method updates the noise level if necessary.

        Parameters
        ----------
        K_d: Tensor
            The (pushforwarded from model) data covariance matrix (stripped
            from sigma0). Should be double.
        sigma0: Tensor
            The (model side) standard deviation.

        Returns
        -------
        Tensor
            Inversion operator R^-1
        float
            When noise has to be increased to make matrices invertible, this
            gives the new value of the noise standard deviation.
        """
        data_std_orig = data_std
        n_data = K_d.shape[0]
        data_ones = torch.eye(n_data, dtype=torch.float64, device=self.device)
        self.R = (data_std**2) * data_ones + self.sigma0**2 * K_d

        # Check condition number if debug mode on.
        if not __debug__:
            self.logger.info(
                    "Condition number of (inverse) inversion operator: {}".format(
                    np.linalg.cond(self.R.cpu().detach().numpy())))

        # Try to invert.
        for attempt in range(50):
            try:
                inversion_op = torch.inverse(self.R)
            except RuntimeError:
                print("Inversion failed: Singular Matrix.")
                # Increase noise in steps of 5%.
                data_std += 0.05 * data_std
                self.R = (data_std**2) * data_ones + self.sigma0**2 * K_d
                print("Increasing data std from original {} to {} and retrying.".format(
                        data_std_orig, data_std))
            else:
                return inversion_op, data_std
        # If didnt manage to invert.
        raise ValueError(
            "Impossible to invert matrix, even at noise std {}".format(self.data_std))
        return -1, data_std

    # TODO: DEPRECATED.
    def get_inversion_op_cholesky(self, K_d, data_std):
        """ Compute the Cholesky decomposition of the inverse of the inversion operator.
        Increases noise level if necessary to make matrix invertible.

        Note that this method updates the noise level if necessary.

        Parameters
        ----------
        K_d: Tensor
            The (pushforwarded from model) data covariance matrix (stripped
            from sigma0).
        sigma0: Tensor
            The (model side) standard deviation.

        Returns
        -------
        Tensor
            L such that R = LL^t. L is lower triangular.
        float
            When noise has to be increased to make matrices invertible, this
            gives the new value of the noise standard deviation.
        """
        raise ValueError("Warning, this is deprecated since we stopped using Cholesky for inversion.")

        data_std_orig = data_std
        n_data = K_d.shape[0]
        data_ones = torch.eye(n_data, dtype=torch.float32, device=self.device)
        self.R = (data_std**2) * data_ones + self.sigma0**2 * K_d

        # Check condition number if debug mode on.
        if not __debug__:
            self.logger.info(
                    "Condition number of (inverse) inversion operator: {}".format(
                    np.linalg.cond(self.R.cpu().detach().numpy())))

        # Try to Cholesky.
        for attempt in range(50):
            try:
                L = torch.cholesky(self.R)
            except RuntimeError:
                self.logger.info("Cholesky failed: Singular Matrix.")
                # Increase noise in steps of 5%.
                data_std += 0.05 * data_std
                self.R = (data_std**2) * data_ones + self.sigma0**2 * K_d
                self.logger.info("Increasing data std from original {} to {} and retrying.".format(
                        data_std_orig, data_std))
            else:
                return L, data_std
        # If didnt manage to invert.
        raise ValueError(
            "Impossible to invert matrix, even at noise std {}".format(self.data_std))
        return -1, data_std
    
    # TODO: DEPRECATED since deprecation of Cholesky.
    def inv_op_vector_mult(self, x):
        """ Multiply a vector by the inversion operator, using Cholesky
        approach (numerically more stable than computing inverse).

        Parameters
        ----------
        x: Tensor
            The vector to multiply.

        Returns
        -------
        Tensor
            Multiplied vector R^(-1) * x.

        """
        raise ValueError("Warning, this is deprecated since we stopped using Cholesky for inversion.")

        z, _ = torch.triangular_solve(x, self.inv_op_L, upper=False)
        y, _ = torch.triangular_solve(z, self.inv_op_L.t(), upper=True)
        return y

    def sample_prior(self):
        """ Sample from prior model.

        Returns
        -------
        sample: (self.n_cells, 1) Tensor
            Column vector of sampled values at each cells.

        """
        import volcapy.covariance.sample as Rinterface
        m0 = self.m0.detach().cpu().item()
        prior = m0 * torch.ones(self.cells_coords.shape[0])
        sigma0 = self.sigma0.detach().cpu().item()
        lambda0 = self.lambda0

        sample = Rinterface.sample(self.kernel, sigma0, lambda0, m0, self.cells_coords)
        return sample

    def posterior_variance(self):
        """ Computes posterior variance at every point.
        WARNING: needs pushforward and inversion operator to be only computed,
        hence, only call after having conditioned the model.

        Returns
        -------
        variances: (self.cells_coords.n_cells) Tensor
            Variance at each point.

        """
        prior_variances = self.kernel.compute_diagonal(
                self.lambda0, self.cells_coords, self.device,
                n_chunks=self.n_chunks, n_flush=50)

        prior_variances -= self.sigma0**2 * torch.einsum("ij,jk,ik->i",
                self.pushfwd.double(), self.inversion_operator,
                self.pushfwd.double())

        self.post_variance = self.sigma0**2 * prior_variances
        return self.post_variance

    def prior_covariance(self):
        """ Computes prior covariance matrix.

        Returns
        -------
        cov_mat: (self.cells_coords.n_cells, self.cells_coords.n_cells) Tensor
            Posterior covariance matrix.

        """
        prior_covariance = self.kernel.compute_full_cov(
                self.lambda0, self.cells_coords, self.device,
                n_chunks=self.n_chunks, n_flush=50)

        return self.sigma0**2 * prior_covariance

    def posterior_covariance(self):
        """ Computes posterior covariance matrix.
        WARNING: needs pushforward and inversion operator to be only computed,
        hence, only call after having conditioned the model.

        Returns
        -------
        cov_mat: (self.cells_coords.n_cells, self.cells_coords.n_cells) Tensor
            Posterior covariance matrix.

        """
        prior_covariance = self.kernel.compute_full_cov(
                self.lambda0, self.cells_coords, self.device,
                n_chunks=self.n_chunks, n_flush=50)

        prior_covariance -= self.sigma0**2 * (self.pushfwd.double() @ self.inversion_operator 
                @ self.pushfwd.double().t())

        return self.sigma0**2 * prior_covariance

    def sample_posterior(self, G, y, data_std):
        """ Sample from the posterior using residual kriging.
        
        Parameters
        ----------
        G: tensor
            Measurement matrix
        y: (n_data, 1) Tensor
            Observed data. Has to be column vector.
        data_std: flot
            Data noise standard deviation.

        Returns
        -------
        sample: (self.n_cells, 1) Tensor
            Column vector of sampled values at each cells.

        """
        post_mean, _ = self.condition_model(G, y, data_std, hypothetical=True)

        # Sample from prior.
        sample = self.sample_prior()
        print(sample.shape)

        # Generate hypotherical data.
        noise = torch.normal(mean=0.0, std=data_std, size=(G.shape[0], 1))
        y_prime = G @ sample + noise

        mu_prime, _ = self.condition_model(G, y_prime, data_std, hypothetical=True)

        post_sample = post_mean + sample - mu_prime
        return post_sample

    def update_sample(self, prior_sample, G, y, data_std):
        """ Sample from the posterior by updating a realization from the prior
        using residual kriging.
        This is useful on platforms whe prior sampling doesn't work (usually
        because the R interface is not available). Then prior realizations may
        be generated offline and updated later.
        
        Parameters
        ----------
        prior_sample: (self.n_cells, 1) Tensor
            Prior realization in the form of a 
            column vector of sampled values at each cells.
        G: tensor
            Measurement matrix
        y: (n_data, 1) Tensor
            Observed data. Has to be column vector.
        data_std: flot
            Data noise standard deviation.

        Returns
        -------
        sample: (self.n_cells, 1) Tensor
            Column vector of sampled values at each cells.

        """
        post_mean, _ = self.condition_model(G, y, data_std, hypothetical=True)

        # Generate hypotherical data.
        noise = torch.normal(mean=0.0, std=data_std, size=(G.shape[0], 1))
        y_prime = G @ prior_sample + noise

        mu_prime, _ = self.condition_model(G, y_prime, data_std, hypothetical=True)

        post_sample = post_mean.detach().cpu() + prior_sample - mu_prime.detach().cpu()
        return post_sample

    def coverage(self, lower=None, upper=None):
        """ Compute (current) coverage function.
        Parameters
        ----------
        lower: float, defaults to None
            Lower threshold of the excursion set. If None, then infinity will
            be used.
        upper: float, defaults to None
            Upper threshold of the excursion set. If None, then infinity will
            be used.
    
        Returns
        -------
        coverage: (self.n_cells, 1) Tensor
            Value of the covaerage function (i.e. current excursion probability) 
            at every cell.

        """
        if lower is not None:
            lower = torch.tensor([lower])
        if upper is not None:
            upper = torch.tensor([upper])

        coverage = gaussian_cdf(self.m_post_m, self.post_variance.reshape(-1, 1),
                lower=lower, upper=upper)
        return coverage
