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
from volcapy.utils import _make_column_vector

# General torch settings and devices.
torch.set_num_threads(8)

# Select gpu if available and fallback to cpu else.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# TODO: Refactor to automatically determine in the covariance module.
N_CHUNKS = 100
N_CHUNKS_VAR = 10 # Chunking for the variance extraction and IVR.

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
    def __init__(self, cov_module, lambda0, sigma0, cells_coords):
        """ Build an updatable covariance from a traditional covariance module.

        Params
        ------
        cov_module: CovarianceModule
            Defines which kernel to use.
        lambda0: float
            Prior lengthscale parameter for the module.
        sigma0: float
            Prior standard deviation for the covariance.
        cells_coords: Tensor
            Coordinates of the model points.

        """
        self.cov_module = cov_module
        self.lambda0 = lambda0
        self.sigma0 = sigma0
        self.cells_coords = cells_coords
        self.n_model = cells_coords.shape[0]

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
        F: Tensor
            Measurement operator.
        data_std: float
            Standard deviation of data noise, assumed to be iid centered gaussian.

        """
        self.pushforwards.append(self.mul_right(G.t()))

        # Get inversion op by Cholesky.
        R = G @ self.pushforwards[-1]
        R =  R + data_std**2 * torch.eye(G.shape[0])
        try:
            L = torch.cholesky(R)
        except RuntimeError:
            print("Error inverting.")

        inversion_op = torch.cholesky_inverse(L)
        self.inversion_ops.append(inversion_op)

    def compute_prior_pushfwd(self, G):
        """ Given an operator G, compute the covariance pushforward K_0 G^T,
        i.e. the pushforward with respect to the prior.

        Parameters
        ----------
        G: (n_data, self.n_model) Tensor

        Returns
        -------
        pushfwd: (self.n_model, n_data) Tensor

        """
        pushfwd = self.sigma0**2 * self.cov_module.compute_cov_pushforward(
                self.lambda0, G, self.cells_coords, DEVICE,
                n_chunks=N_CHUNKS,
                n_flush=50)
        return pushfwd

    def extract_variance(self, n_chunks=N_CHUNKS_VAR):
        """ Extracts the pointwise variance from an UpdatableCovariane module.
    
        Parameters
        ----------
        n_chunks: int
            Number of chunks to break the covariane matrix in.
            Increase if computations do not fit in memory.
    
        Returns
        -------
        variances: (cov_module.n_model) Tensor
            Variance at each point.
    
        """
        prior_variances = self.sigma0**2 * self.cov_module.compute_diagonal(
                self.lambda0, self.cells_coords, DEVICE,
                n_chunks=N_CHUNKS, n_flush=50)

        for p, r in zip(self.pushforwards, self.inversion_ops):
            prior_variances -= torch.einsum("ij,jk,ik->i",p,r,p)

        return prior_variances

    def compute_IVR(self, G, data_std, n_chunks=N_CHUNKS_VAR):
        """ Compute the (integrated) variance reduction (IVR) that would
        result from collecting the data described by the measurement operator
        G.    

        Parameters
        ----------
        G: (n_data, self.n_model) Tensor
            Measurement operator
        data_std: float
            Measurement noise standard deviation, assumed to be iid centered
            gaussian.
        n_chunks: int
            Number of chunks to break the covariane matrix in.
            Increase if computations do not fit in memory.
    
        Returns
        -------
        IVR: float
            Integrated variance reduction resulting from the observation of G.    

        """
        # First subdivide the cells in subgroups.
        chunked_indices = torch.chunk(torch.tensor(list(range(self.n_model))), n_chunks)

        # Compute the current pushforward.
        G_dash = self.mul_right(G.t())

        # Get inversion op by Cholesky.
        R = G @ G_dash + data_std**2 * torch.eye(G.shape[0])
        try:
            L = torch.cholesky(R)
        except RuntimeError:
            print("Error inverting.")
        inversion_op = torch.cholesky_inverse(L)

        IVR = 0
        for inds in chunked_indices:
            G_part = G_dash[inds,:]
            V = G_part @ inversion_op @ G_part.t()
            IVR += torch.sum(V.diag())
        return IVR.item()

    def compute_VR(self, G, data_std, n_chunks=N_CHUNKS_VAR):
        """ Compute the variance reduction (VR) (no integral) that would
        result from collecting the data described by the measurement operator
        G.    

        Parameters
        ----------
        G: (n_data, self.n_model) Tensor
            Measurement operator
        data_std: float
            Measurement noise standard deviation, assumed to be iid centered
            gaussian.
        n_chunks: int
            Number of chunks to break the covariane matrix in.
            Increase if computations do not fit in memory.
    
        Returns
        -------
        VR: (self.n_model) Tensor
            Variance reduction, at each point resulting from the observation of G.    

        """
        # First subdivide the model cells in subgroups.
        chunked_indices = torch.chunk(torch.tensor(list(range(self.n_model))), n_chunks)

        # Compute the current pushforward.
        G_dash = self.mul_right(G.t())

        # Get inversion op by Cholesky.
        R = G @ G_dash + data_std**2 * torch.eye(G.shape[0])
        try:
            L = torch.cholesky(R)
        except RuntimeError:
            print("Error inverting.")
        inversion_op = torch.cholesky_inverse(L)

        VR = torch.zeros(self.n_model)
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

        self.n_model = prior.shape[0]
        self.cov_module = cov_module
        
        if not (self.n_model == cov_module.n_model):
            raise ValueError(
                "Model size for mean: {} does not agree with "\
                "model size for covariance {}.".format(
                        self.n_model, cov_module.n_model))
        
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
