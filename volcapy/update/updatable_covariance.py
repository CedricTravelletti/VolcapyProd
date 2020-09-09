""" Make covariance matrix sequentially updatable.

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

TODO
----

Implement some shape getter.
Implement computation of the diagonal.

"""
import torch
import numpy as np
from volcapy.utils import _make_column_vector
from volcapy.gaussian_cdf import gaussian_cdf
from rpy2.robjects import numpy2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# Activate auto-wrapping of numpy arrays as rpy2 objects.
numpy2ri.activate()

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
    def __init__(self, cov_module, lambda0, sigma0, cells_coords, n_chunks):
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

        """
        self.cov_module = cov_module
        self.lambda0 = lambda0
        self.sigma0 = sigma0
        self.cells_coords = cells_coords
        self.n_cells = cells_coords.shape[0]
        self.n_chunks = n_chunks

        self.pushforwards = []
        self.inversion_ops = []

    def mul_right(self, A):
        """ Multiply covariance matrix from the right.
        Note that the matrix with which we multiply should map to a smaller
        dimensional space.

        Parameters
        ----------
        A: Tensor
            Matrix with which to multiply.

        Returns
        -------
        Tensor
            K * A

        """
        # First compute the level 0 pushforward.
        # Warning: the original covariance pushforward method was used to
        # comput K G^T, taking G as an argument, i.e. it does transposing in
        # the background. We hence have to feed it A.t.
        cov_pushfwd_0 = self.compute_prior_pushfwd(A.t())

        for p, r in zip(self.pushforwards, self.inversion_ops):
            cov_pushfwd_0 -= p @ (r @ (p.t() @ A))

        # Note the first term (the one with C_0 alone) only has one sigma0**2
        # factor associated with it, whereas all other terms in the updating
        # have two one.
        return cov_pushfwd_0

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
        self.pushforwards.append(self.mul_right(G.t()))

        # Get inversion op by Cholesky.
        R = G @ self.pushforwards[-1]

        L, _ = self._cholesky_helper(R, data_std)

        inversion_op = torch.cholesky_inverse(L)
        self.inversion_ops.append(inversion_op)

    def _cholesky_helper(self, R, data_std):
        data_std_orig = data_std
        # Try to Cholesky.
        MAX_ATTEMPTS = 200
        for attempt in range(MAX_ATTEMPTS):
            try:
                L = torch.cholesky(R + data_std**2 * torch.eye(R.shape[0]))
            except RuntimeError:
                print("Cholesky failed: Singular Matrix.")
                # Increase noise in steps of 5%.
                data_std += 0.05 * data_std
                print(
                        "Increasing data std from original {} to {} and retrying.".format(
                        data_std_orig, data_std))
            else:
                return L, data_std
        # If didnt manage to invert.
        raise ValueError(
            "Impossible to invert matrix, even at noise std {}".format(self.data_std))
        return -1, data_std

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
        pushfwd = self.sigma0**2 * self.cov_module.compute_cov_pushforward(
                self.lambda0, G, self.cells_coords, DEVICE,
                n_chunks=self.n_chunks,
                n_flush=50)
        return pushfwd

    def extract_variance(self):
        """ Extracts the pointwise variance from an UpdatableCovariane module.
    
        Returns
        -------
        variances: (cov_module.n_cells) Tensor
            Variance at each point.
    
        """
        prior_variances = self.sigma0**2 * self.cov_module.compute_diagonal(
                self.lambda0, self.cells_coords, DEVICE,
                n_chunks=self.n_chunks, n_flush=50)

        for p, r in zip(self.pushforwards, self.inversion_ops):
            prior_variances -= torch.einsum("ij,jk,ik->i",p,r,p)

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

        # Get inversion op by Cholesky.
        R = G @ G_dash

        L, _ = self._cholesky_helper(R, data_std)

        inversion_op = torch.cholesky_inverse(L)

        IVR = 0
        if weights is None:
            for inds in chunked_indices:
                G_part = G_dash[inds,:]
                V = G_part @ inversion_op @ G_part.t()
                IVR += torch.sum(V.diag())
        else:
            for inds in chunked_indices:
                G_part = G_dash[inds,:]
                V = G_part @ inversion_op @ G_part.t()
                IVR += torch.sum(V.diag() * weights[inds])
        return IVR.item()

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

        """
        # First subdivide the model cells in subgroups.
        chunked_indices = torch.chunk(torch.tensor(list(range(self.n_cells))),
                self.n_chunks)

        # Compute the current pushforward.
        G_dash = self.mul_right(G.t())

        # Get inversion op by Cholesky.
        R = G @ G_dash + data_std**2 * torch.eye(G.shape[0])
        try:
            L = torch.cholesky(R)
        except RuntimeError:
            print("Error inverting.")
        inversion_op = torch.cholesky_inverse(L)

        VR = torch.zeros(self.n_cells)
        for inds in chunked_indices:
            G_part = G_dash[inds,:]
            V = G_part @ inversion_op @ G_part.t()
            VR[inds] = V.diag()
        return VR
            

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
        self.m = self.m + K_dash @ R @ (y - G @ self.m)

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
    def __init__(self, cov_module, lambda0, sigma0, m0, cells_coords):
        self.covariance = UpdatableCovariance(cov_module, lambda0,
                sigma0, cells_coords, n_chunks)
        self.mean = UpdatableMean(m0 * torch.ones(cells_coords.shape[0]),
            self.covariance)

        self.n_cells = cells_coords.shape[0]

    @property
    def mean_vec(self):
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
        # Import the R RandomFields library.
        rflib = importr("RandomFields")

        # Create the model and sample.
        # TODO: Autodetect covariance model.
        model = rflib.RMmatern(nu=2.5, var=self.covariance.sigma0**2,
                scale=self.covariance.lambda0)
        simu = rflib.RFsimulate(model,
                self.covariance.cells_coords.detach().cpu().numpy())

        # Back to numpy, make column vector also.
        sample = torch.from_numpy(
                np.asarray(simu.slots['data']).astype(np.float32)[:, None])

        return sample + self.mean.m

    def sample_conditional(self, G, y, data_std):
        """ Sample conditionally on the data GZ = y.
        Note that we here only condition on that data, starting from the prior,
        i.e. there are no update involved.

        """
        prior_sample = self.sample_prior

        # Update model.
        self.update(G, y, data_std)
        return self.prior_mean_vec + self.mean_vec

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
        return self.covariance.IVR(G, data_std, integration_inds, weights)

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
        variance = self.covariance..extract_variance()
        mean = self.mean_vec

        if lower is not None:
            lower = torch.tensor([lower])
        if upper is not None:
            upper = torch.tensor([upper])

        weights = gaussian_cdf(mean, variance, lower=lower, upper=upper)
        return self.IVR(G, data_std, weights=weights)
