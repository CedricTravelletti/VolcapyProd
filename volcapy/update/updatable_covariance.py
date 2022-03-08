""" VERSION WHERE WE DO NOT TRY TO OPTIMIZE SIGMA.

Make covariance matrix sequentially updatable.

The goal is that, instead of performing a conditioning on lots of measurement
in one go, we can instead chunk the data and perform several conditioning in
series, updating the covariance matrix along the way.

The CRUX of this module is that it handles covariance matrices that are too
large to fit in memory.

This is defined in the document "05-12-2019: cov_update_implementation".

Remark: Someday, we might want to switch to KeOps for the covariance matrix.

Concept
-------
Let F_1,..., F_n be measurement operators/forwards.
We want to compute the conditional covariance corresponding to those
measurements. We could stack the matrices an just condition on one big
measurement/forward. This would require us to inverse a matrix the size of the
whole dataset.

Alsok we work in the *big model* framework, i.e. when the model discretization
is too fine to allow covariance matrices to ever sit in memory (contrast this
with the *big data* settting.

Testing scripts may be found in Volcano/tests/test_update.py.

DTYPE
-----
We store inversion operators in double.

TODO
----

Implement some shape getter.
Implement computation of the diagonal.

REFACTOn USING OBSERVER PATTERN: UpdatableMean, Realizations and so on register
themselves to the cov module.


"""
import torch
import numpy as np
import pickle
import warnings
import time
from volcapy.utils import _make_column_vector
from volcapy.gaussian_cdf import gaussian_cdf


# General torch settings and devices.
torch.set_num_threads(8)

# Select gpu if available and fallback to cpu else.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class UpdatableCovariance:
    """ Covariance matrix that can be sequentially updated to include new
    measurements (conditioning).

    Attributes
    ----------
    pushforwards: List[Tensor]
        The covariance pushforwards corresponding to each conditioning step.
    inversion_ops: List[Tensor]
        The inversion operators corresponding to each conditioning step.

    """
    def __init__(self, cov_module, lambda0, sigma0, cells_coords,
            n_chunks=200, n_flush=50):
        """ Build an updatable covariance from a traditional covariance module.

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
        n_chunks: int
            Number of chunks to break the covariane matrix in.
            Increase if computations do not fit in memory.
        n_flush: int
            Synchronize threads and flush GPU cache every *n_flush* iterations.
            This is necessary to avoid OOM errors.
            Default is 50.

        """
        self.cov_module = cov_module
        if not torch.is_tensor(lambda0):
            lambda0 = torch.tensor([lambda0])
        self.lambda0 = lambda0.to(DEVICE)
        if not torch.is_tensor(sigma0):
                sigma0 = torch.tensor([sigma0])
        self.sigma0 = sigma0.to(DEVICE)
        
        if not torch.is_tensor(cells_coords):
            cells_coords = torch.from_numpy(cells_coords)
        self.cells_coords = cells_coords.to(DEVICE)
        self.n_cells = cells_coords.shape[0]

        self.n_chunks = n_chunks
        self.n_flush = n_flush

        self.pushforwards = []
        self.inversion_ops = []

    def mul_right(self, A, strip=False):
        """ Multiply covariance matrix from the right.
        Note that the matrix with which we multiply should map to a smaller
        dimensional space.

        Also provides a stripped version, where the sigma0**2 factor is removed
        (can be used to improve numerical precision).

        Note that results are returned in double precision.

        Parameters
        ----------
        A: Tensor
            Matrix with which to multiply.
        strip: bool, defaults to False
            If true, then will return 1/sigma0**2 K * A.

        Returns
        -------
        Tensor
            K * A

        """
        start = time.time()
        # If both devices not equal, fallback to standard device.
        if not A.device == DEVICE: A = A.to(DEVICE)

        # First compute the level 0 pushforward.
        # Warning: the original covariance pushforward method was used to
        # comput K G^T, taking G as an argument, i.e. it does transposing in
        # the background. We hence have to feed it A.t.
        cov_pushfwd_0 = self.compute_prior_pushfwd(A.t()).double()

        temp = torch.zeros(cov_pushfwd_0.shape, dtype=torch.float64)

        for p, r in zip(self.pushforwards, self.inversion_ops):
            temp -= p.double() @ (r @ (p.double().t() @ A.double()))

        end = time.time()
        print((end - start) / 60.0)
        torch.cuda.empty_cache()
        # Note the first term (the one with C_0 alone) only has one sigma0**2
        # factor associated with it, whereas all other terms in the updating
        # have two one.
        if strip:
            return cov_pushfwd_0 + temp.to(DEVICE)
        else:
            return cov_pushfwd_0 + temp.to(DEVICE)

    # TODO: DEPRECATED. Not adapted to new stripped pushforwards.
    # TODO: Is this ever used?
    def sandwich(self, A):
        """ Sandwich the covariance matrix on both sides.
        Note that the matrix with which we sandwich should map to a smaller
        dimensional space.

        Parameters
        ----------
        A: Tensor
            Matrix with which to sandwich.

        Returns
        -------
        Tensor
            A^t * C * A

        """
        raise NotImplementedError

        # First compute the level 0 pushforward, i.e. K_0 A.
        # Warning: the original covariance pushforward method was used to
        # comput K G^T, taking G as an argument, i.e. it does transposing in
        # the background. We hence have to feed it A.t.
        cov_pushfwd_0 = A.t() @ self.compute_prior_pushfwd(A.t())

        for p, r in zip(self.pushforwards, self.inversion_ops):
            tmp = p.t() @ A
            cov_pushfwd_0 -= tmp.t() @ (r @ tmp)

        return cov_pushfwd_0

    def update(self, G, data_std):
        """ Update the covariance matrix / perform a conditioning.

        Params
        ------
        G: Tensor
            Measurement operator.
        data_std: float
            Standard deviation of data noise, assumed to be iid centered gaussian.

        """
        # Pushforwards are big, so store in single precision.
        # But keep in double for the inversion operator.
        current_pushfwd = self.mul_right(G.t(), strip=True)
        self.pushforwards.append(current_pushfwd.float())

        # Get inversion op.
        K_d = G.double() @ current_pushfwd
        inversion_op, _ = self._inversion_helper(K_d, data_std)

        self.inversion_ops.append(inversion_op)

    def _inversion_helper(self, K_d, data_std):
        data_std_orig = data_std
        # Try to invert.
        MAX_ATTEMPTS = 200
        for attempt in range(MAX_ATTEMPTS):
            try:
                inversion_op = torch.inverse(K_d + data_std *
                        torch.eye(K_d.shape[0], dtype=torch.float64))
            except RuntimeError:
                print("Inversion failed: Singular Matrix.")
                # Increase noise in steps of 5%.
                data_std += 0.05 * data_std
                print(
                        "Increasing data std from original {} to {} and retrying.".format(
                        data_std_orig, data_std))
            else:
                return inversion_op.double(), data_std
        # If didnt manage to invert.
        raise ValueError(
            "Impossible to invert matrix, even at noise std {}".format(self.data_std))
        return -1, data_std

    def compute_prior_pushfwd(self, G, lambda0=None, sigma0=None):
        """ Given an operator G, compute the covariance pushforward K_0 G^T,
        i.e. the pushforward with respect to the prior.

        Parameters
        ----------
        G: (n_data, self.n_cells) Tensor
        lambda0: Tensor, defaults to None
            If not provided, use self.lambda0 as lengthscale.

        Returns
        -------
        pushfwd: (self.n_cells, n_data) Tensor

        """
        # If both devices not equal, fallback to standard device.
        if not G.device == DEVICE: G = G.to(DEVICE)

        if lambda0 is None: lambda0 = self.lambda0
        if sigma0 is None: lambda0 = self.sigma0
        sigma0 = sigma0.to(DEVICE)

        pushfwd = self.cov_module.compute_cov_pushforward(
                lambda0, G, self.cells_coords, DEVICE,
                n_chunks=self.n_chunks,
                n_flush=self.n_flush)
        return sigma0**2 * pushfwd

    def extract_variance(self):
        """ Extracts the pointwise variance from an UpdatableCovariance module. 
        I.e. extracts the diagonal.
    
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

        return prior_variances

    def IVR(self, G, data_std, integration_inds=None, weights=None):
        """ Compute the (integrated) variance reduction (IVR) that would
        result from collecting the data described by the measurement operator
        G.    

        Parameters
        ----------
        G: (n_data, self.n_cells) Tensor
            Measurement operator
        data_std: float
            Measurement noise standard deviation, assumed to be iid centered
            gaussian.
        integration_inds: array_like [int]
            List of indices (wrt the model grid) over which to integrate. May
            be used if only want to consider some region. Defaults to the whole
            grid.
        weights: (self.n_cells) Tensor, optional
            Can be provided to weight the integral differently for each cell.
    
        Returns
        -------
        IVR: float
            Integrated variance reduction resulting from the observation of G.    

        """
        if integration_inds is None:
            integration_inds = list(range(self.n_cells))

        # First subdivide the cells in subgroups.
        chunked_indices = torch.chunk(torch.tensor(integration_inds).long(),
                self.n_chunks)

        # Compute the current pushforward.
        G_dash = self.mul_right(G.t())


        # Get inversion op.
        R = G.double() @ G_dash.double()
        inversion_op, _ = self._inversion_helper(R, data_std)

        IVR = 0
        if weights is None:
            for inds in chunked_indices:
                G_part = G_dash[inds,:]
                V = G_part.float() @ inversion_op.float() @ G_part.t().float()
                IVR += torch.sum(V.diag())
        else:
            for inds in chunked_indices:
                G_part = G_dash[inds,:]
                V = G_part.float() @ inversion_op.float() @ G_part.t().float()
                IVR += torch.sum(V.diag() * weights[inds])
        return IVR.item()

    def compute_fantasy_variance(self, G, data_std):
        """ Compute the posterior variance that would
        result from collecting the data described by the measurement operator
        G.    

        Parameters
        ----------
        G: (n_data, self.n_cells) Tensor
            Measurement operator
        data_std: float
            Measurement noise standard deviation, assumed to be iid centered
            gaussian.
    
        Returns
        -------
        variance: (self.n_cells) Tensor
            Variance at each point resulting from the observation of G.    
        G_dash: (self.n_cells, G.shape[0]) Tensor
            Covariance pushforward for the considered potential observation.

        """
        # Compute variance reduction.
        VR, G_dash = self.compute_VR(G, data_std)
        variance = self.extract_variance()

        return (variance - VR, G_dash)

    def compute_VR(self, G, data_std):
        """ Compute the variance reduction (VR) (no integral) that would
        result from collecting the data described by the measurement operator
        G.    

        Parameters
        ----------
        G: (n_data, self.n_cells) Tensor
            Measurement operator
        data_std: float
            Measurement noise standard deviation, assumed to be iid centered
            gaussian.
    
        Returns
        -------
        VR: (self.n_cells) Tensor
            Variance reduction, at each point resulting from the observation of G.    
        G_dash: (self.n_cells, G.shape[0]) Tensor
            Covariance pushforward for the considered potential observation.

        """
        # First subdivide the model cells in subgroups.
        chunked_indices = torch.chunk(torch.tensor(list(range(self.n_cells))),
                self.n_chunks)

        # Compute the current pushforward.
        G_dash = self.mul_right(G.t())

        # Get inversion op.
        R = G.double() @ G_dash.double()
        inversion_op, _ = self._inversion_helper(R, data_std)

        VR = torch.zeros(self.n_cells)
        for inds in chunked_indices:
            G_part = G_dash[inds,:]
            V = G_part.float() @ inversion_op.float() @ G_part.t().float()
            VR[inds] = V.diag()
        return (VR, G_dash)

    def condition_fantasy_data(self, prior, stacked_G, fantasy_ys,
            splitted_inds):
        """ Compute the posterior mean that would result from observing other
        data than the one which was assimilated (over the full history).
        
        The main use of this method is to compute conditional realizations
        using residual kriging.

        Note that this method uses the pre-computed intermediate inversion
        operators, hence each datapoint should correspond to one of the
        observation operators used in the update and the noise level should be
        the same (since it enters the computation of the inversion operators).

        Parameters
        ----------
        prior: (n_cells, 1) Tensor
            Vector defining the prior mean at the model points.
        stacked_G: (d, n_cells) Tensor
            Observation operators at each step (stacked). Assumed to correspond
            to the ones used to update the GP.
            WARNING: Will only work if the GP has been updated with one observation at a time.
        fantasy_ys: (len(self.pushforwards)) Tensor
            Vector of observed data values. The n-element should correspond to
            the n-th updating operation which was performed, hence to the n-th
            observation operator.
            Note this is a bit awkward, since currently the
            UpdatableCovariance module does not inform the user about which
            observation operator was used at stage n.
            Moreover, this procedure currently only allows for 1-datapoint
            observations.

        Returns
        -------
        conditional_mean: (self.n_cells, 1) Tensor
            Conditional mean, conditional on the provided data.

        """
        conditional_mean = prior.double()
        for i, inds in enumerate(splitted_inds):
            y = _make_column_vector(fantasy_ys[inds]).double()
            K_dash = self.pushforwards[i].double()
            R = self.inversion_ops[i]
            conditional_mean = (
                    conditional_mean.double()
                    + K_dash @ R @ 
                    (y - stacked_G[inds, :].double() @ conditional_mean).double())
        return conditional_mean.float()

    def __dict__(self):
        state_dict = {
                'kernel_family': self.cov_module.KERNEL_FAMILY,
                'lambda0': self.lambda0,
                'sigma0': self.sigma0,
                'cells_coords': self.cells_coords.cpu().numpy(),
                'n_chunks': self.n_chunks,
                'pushforwards': [p.cpu().numpy() for p in self.pushforwards],
                'inversion_ops': [inv.cpu().numpy() for inv in self.inversion_ops]
                }
        return state_dict
            

class UpdatableMean:
    """ Mean vector that can be sequentially updated. This is an addon to the
    UpdatableCovariance class in that it depends on such an object.

    """
    def __init__(self, prior, cov_module):
        """ Build an updatable mean.

        Params
        ------
        prior: (n_cells, 1) Tensor
            Vector defining the prior mean at the model points.
            We require the number of points to be the same as in the updatable
            covariance module. Note that we want a column vector.
        cov_module: UpdatableCovariance

        """
        prior = _make_column_vector(prior)

        self.prior = prior
        self.m = prior # Current value of conditional mean.

        self.n_cells = prior.shape[0]
        self.cov_module = cov_module
        
        if not (self.n_cells == cov_module.n_cells):
            raise ValueError(
                "Model size for mean: {} does not agree with "\
                "model size for covariance {}.".format(
                        self.n_cells, cov_module.n_cells))
        
    # TODO: Find a better design patter. This subtlety of having to make sure
    # that the covariance_module has been updated first is dangerous. Maybe an
    # observer pattern on something like that.
    def update(self, y, G):
        """ Updates the means.
        WARNING: should only be used after the covariance module has been
        updated, since it depends on it having computed the latest quantities.

        Params
        ------
        y: Tensor
            Data vector.
        G: Tensor
            Measurement matrix.
        data_std: float
            Measurement noise standard deviation, assumed to be iid centered
            gaussian.

        """
        y = _make_column_vector(y)

        # Get the latest conditioning operators.
        K_dash = self.cov_module.pushforwards[-1]
        R = self.cov_module.inversion_ops[-1]
        self.m = (self.m.double() 
                + K_dash.double() @ R.double() @ (y - G @
                self.m).double()).float()

    def __dict__(self):
        state_dict = {
                'prior': self.prior.cpu().numpy(),
                'm': self.m.cpu().numpy(),
                }
        return state_dict


# TODO: Should it just inherit from UpdatableMean?
class UpdatableRealization(UpdatableMean):
    """ Posterior conditional realization that can be sequentially updated.

    This class uses residual kriging to produce a sample from the posterior,
    which may be updated to reflect updates to the posterior distribution.
    Residual kriging provides a conditional realzation as a sum of the
    conditional mean, plus a correction that contains a conditional mean,
    conditional on *simulated* data that comes from a prior realization.

    Hence, the module will use the (already computed) conditional mean of some
    UpdatableGP and then *replay* the updates onto a sample from the prior.

    """
    def __init__(self, prior, gp_module):
        """ Build an updatable realization.

        Params
        ------
        prior_realization: (n_cells, 1) Tensor
            Vector defining the prior realization at the model points.
            We require the number of points to be the same as in the updatable
            covariance module. Note that we want a column vector.
        gp_module: UpdatableGP

        """
        super().__init__(prior, gp_module.covariance)
        self.gp = gp_module

    @classmethod
    def bootstrap(cls, prior_realization, stacked_G, data_std, gp_module,
            splitted_inds):
        """ Bootstrap an UpdatableRealization diretly to some posterior state.
        I.e. given a GP module that has already assimilated some data,
        we directly update the realization to that state.

        Parameters
        ----------
        prior_realization: (n_cells, 1) Tensor
            Vector defining the prior realization at the model points.
            We require the number of points to be the same as in the updatable
            covariance module. Note that we want a column vector.
        stacked_G: (d, n_cells) Tensor
            Observation operators at each step (stacked). Assumed to correspond
            to the ones used to update the GP.
            WARNING: Will only work if the GP has been updated with one observation at a time.
        data_std: float
            Measurement noise standard deviation, assumed to be iid centered
            gaussian.
        gp_module: UpdatableGP

        Returns
        -------
        UpdatableRealization
            A conditional realization at the current state of the covariance
            module.

        """
        # Create a non-conditioned realization.
        updatable_realization = cls(prior_realization, gp_module)

        # Bootstrap the unperformed updates.
        noise = torch.normal(mean=0.0, std=data_std, size=(stacked_G.shape[0], 1))
        fantasy_ys = stacked_G @ prior_realization + noise
        updatable_realization.set_conditional_mean(
                gp_module.covariance.condition_fantasy_data(prior_realization,
                        stacked_G, fantasy_ys,
                        splitted_inds))
        return updatable_realization
        
    # TODO: Find a better design patter. This subtlety of having to make sure
    # that the covariance_module has been updated first is dangerous. Maybe an
    # observer pattern or something like that.
    # WARNING: NOIT FINISHED.
    def update(self, G, y, data_std):
        """ Updates the realization.
        WARNING: should only be used after the covariance module has been
        updated, since it depends on it having computed the latest quantities.

        Params
        ------
        y: Tensor
            Data vector.
        G: Tensor
            Measurement matrix.
        data_std: float
            Measurement noise standard deviation, assumed to be iid centered
            gaussian.

        """
        # Compute simulated data.
        noise = torch.normal(mean=0.0, std=data_std, size=(G.shape[0], 1))
        y_prime = G @ self.m + noise

        y = _make_column_vector(y)
        y_prime = _make_column_vector(y_prime).double()

        # Get the latest conditioning operators.
        K_dash = self.cov_module.pushforwards[-1]
        R = self.cov_module.inversion_ops[-1]
        self.m = (self.m.double() 
                + K_dash.double() @ R.double() @ (y - y_prime)).float()


class UpdatableGP():
    """ Bundles the two above classes into an updatable Gaussian process.

    Parameters
    ----------
    cov_module: CovarianceModule
        Defines which kernel to use.
    lambda0: float
        Prior lengthscale parameter for the module.
    sigma0: float
        Prior standard deviation for the covariance.
    m0: float
        Prior mean, constant over the domain.
    cells_coords: (n_cells, n_dims) Tensor
        Coordinates of the model points.
    n_chunks: int
        Number of chunks to break the covariane matrix in.
        Increase if computations do not fit in memory.

    """
    def __init__(self, cov_module, lambda0, sigma0, m0,
            cells_coords, n_chunks):
        self.covariance = UpdatableCovariance(cov_module, lambda0,
                sigma0, cells_coords, n_chunks)
        self.mean = UpdatableMean(m0 * torch.ones(cells_coords.shape[0]),
            self.covariance)

        self.n_cells = cells_coords.shape[0]

    @property
    def mean_vec(self):
        """ Returns the current mean vector.

        Returns
        -------
        mean_vec: Tensor (self.n_cell, 1)

        """
        return self.mean.m

    @property
    def prior_mean_vec(self):
        return self.mean.prior

    # So we only store once.
    @property
    def cells_coords(self):
        return self.covariance.cells_coords

    def update(self, G, y, data_std):
        """ Given some data, update the model.

        Parameters
        ----------
        G: (n_data, self.n_cells) Tensor
            Measurement (forward) operator.
        y: Tensor
            Data vector.
        data_std: float
            Standard deviation of observations noise (assumed centred iid
            gaussian).

        """
        self.covariance.update(G, data_std)
        self.mean.update(y, G)

    def sample_prior(self):
        """ Sample from prior model.

        Returns
        -------
        sample: (self.n_cells, 1)
            Column vector of sampled values at each cells.

        """
        import volcapy.covariance.sample as Rinterface
        sigma0 = self.covariance.sigma0.detach().cpu().item()
        lambda0 = self.covariance.lambda0

        centred_sample = Rinterface.sample(self.covariance.cov_module, sigma0, lambda0,
                0.0, self.cells_coords)
        return centred_sample + self.prior_mean_vec

    # TODO: Finish implementation.
    def sample_conditional(self, G, y, data_std):
        """ Sample conditionally on the data GZ = y.
        Note that we here only condition on that data, starting from the prior,
        i.e. there are no update involved.

        """
        raise NotImplementedError

        prior_sample = self.sample_prior

        # Update model.
        self.update(G, y, data_std)
        return self.prior_mean_vec + self.mean_vec

    def IVR(self, G, data_std, integration_inds=None, weights=None):
        """ Compute the (integrated) variance reduction (IVR) that would
        result from collecting the data described by the measurement operator
        G. Note that this function can also be used to compute the weighted IVR
        criterion by providing the weights argument.

        Parameters
        ----------
        G: (n_data, self.n_cells) Tensor
            Measurement operator
        data_std: float
            Measurement noise standard deviation, assumed to be iid centered
            gaussian.
        integration_inds: array_like [int]
            List of indices (wrt the model grid) over which to integrate. May
            be used if only want to consider some region. Defaults to the whole
            grid.
        weights: (self.n_cells) Tensor, optional
            Can be provided to weight the integral differently for each cell.
    
        Returns
        -------
        IVR: float
            Integrated variance reduction resulting from the observation of G.    

        """
        return self.covariance.IVR(G, data_std, integration_inds, weights)

    # TODO: Deprecated. Should be removed.
    def weighted_IVR(self, G, data_std, lower=None, upper=None):
        """ Weighted IVR crtierion for learning excurions sets. The user can
        specifies the excursion set to recover by providing upper and lower
        thresholds. Thresholds are optional, and if one isn't provided, it will
        default to infinity. The excursion set to recover is defined as the
        region where the GP takes values in the interval [lower, upper].

        Parameters
        ----------
        G: (n_data, self.n_cells) Tensor
            Measurement operator
        data_std: float
            Measurement noise standard deviation, assumed to be iid centered
            gaussian.
        lower: float, defaults to None
            Lower threshold of the excursion set. If None, then infinity will
            be used.
        upper: float, defaults to None
            Upper threshold of the excursion set. If None, then infinity will
            be used.
    
        Returns
        -------
        weighted_IVR: float
            Integrated variance reduction resulting from the observation of G.    

        """
        variance = self.covariance.extract_variance()
        mean = self.mean_vec

        if lower is not None:
            lower = torch.tensor([lower])
        if upper is not None:
            upper = torch.tensor([upper])

        weights = gaussian_cdf(mean, variance.reshape(-1, 1),
                lower=lower, upper=upper)
        return self.IVR(G, data_std, weights=weights)

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
        variance = self.covariance.extract_variance()
        mean = self.mean_vec

        if lower is not None:
            lower = torch.tensor([lower])
        if upper is not None:
            upper = torch.tensor([upper])

        coverage = gaussian_cdf(mean, variance.reshape(-1, 1),
                lower=lower, upper=upper)
        return coverage

    def neg_log_likelihood(self, lambda0, sigma0, m0, G, y, data_std=0.0):
        """ Compute the negative log-likelihood (up to a constant and a factor 1/2).

        Parameters
        ----------
        lambda0: Tensor
            Lengthscale parameter.
        sigma0: Tensor
            Prior standard deviation.
        m0: Tensor
            Prior mean.
        G: Tensor (n_data, n_model)
            Observation operator.
        y: Tensor (n_data)
            The data vector.

        Returns
        -------
        neg_log_likelihood: Tensor

        """
        y = y.reshape(-1, 1)
        prior_mean = m0 * torch.ones(G.shape[1], 1)

        pushfwd = self.covariance.compute_prior_pushfwd(G, lambda0, sigma0).cpu()
        data_cov = G @ pushfwd + data_std**2 * torch.eye(G.shape[0])

        nll = (torch.logdet(data_cov)
                + (y - G @ prior_mean).t() 
                @ torch.inverse(data_cov) 
                @ (y - G @ prior_mean))
        return nll

    def __dict__(self):
        return {'mean': self.mean.__dict__(),
                'covariance': self.covariance.__dict__()}

    def save(self, path):
        """ Saves current state to file.

        Parameters
        ----------
        path: string
            Where to save the state.

        """
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__(), f)
            f.close()

    @classmethod
    def load(cls, path):
        """ Restore the state of an updatable GP from file.

        Parameters
        ----------
        path: string
            Path to file where state is saved.

        Returns
        -------
        UpdatableGP
            An UpdatableGP with state restored from file, i.e. we do not need
            to recompute the intermediate quantities and directrly has access
            to the conditional GP.

        """
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)
            f.close()

        # Load the correct kernel module.
        import importlib
        cov_module = importlib.import_module(
                "volcapy.covariance.{}".format(state_dict['covariance']['kernel_family']))

        # Create the bare GP.
        gp = cls(cov_module,
                state_dict['covariance']['lambda0'],
                state_dict['covariance']['sigma0'],
                # TODO: This is not clean. Fix it.
                state_dict['mean']['prior'][0].item(),
                state_dict['covariance']['cells_coords'],
                state_dict['covariance']['n_chunks'])

        # Restore the GP mean and covariance module.
        gp.covariance.pushforwards = [torch.from_numpy(x) for x in
                state_dict['covariance']['pushforwards']]
        gp.covariance.inversion_ops = [torch.from_numpy(x) for x in
                state_dict['covariance']['inversion_ops']]
        gp.mean.m = torch.from_numpy(state_dict['mean']['m'])

        return gp

    # TO implement.
    def rewind(self, step):
        """ Go back to a certain step.
        
        """
        self.covariance.pushforwards = self.covariance.pushforwards[:step]
        self.covariance.inversion_ops = self.covariance.inversion_ops[:step]

        # self.mean.m = 
