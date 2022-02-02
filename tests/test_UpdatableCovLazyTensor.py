""" Test the conjugate gradient implementation of GPytorch.

The CG implementation of GPytorch allows one to compute A^{-1}b by only 
providing a multiplication routine for the matrix A. 
On top of that, their implementation works for batch right-hand side and 
is also able to return Lanczos tridiagonalizations of the A matrix. 

References:
    - https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/utils/linear_cg.py
    - https://arxiv.org/pdf/1809.11165.pdf

"""
import torch
from volcapy.update.updatable_covariance import UpdatableGP
from volcapy.update.lazy_updatable_covariance import UpdatableCovLazyTensor
import volcapy.covariance.matern52 as kernel
from volcapy.inverse.toy_fourier_2d import ToyFourier2d
from gpytorch.lazy import MatmulLazyTensor
from gpytorch.utils import linear_cg


# Here test the conjugate gradient inversion from gpytorch.
n = 100
A = torch.eye(n)
rhs = torch.rand((n, 1))

def matmul_closure(x):
    return A @ x

ans = linear_cg(matmul_closure, rhs)


# Test the LazyTensor wrapper for UpdatableCovariance. 
# In particular test the pivoted Cholesky decomposition.
m0 = 1.0
sigma0 = 2.0
lambda0 = 0.5

n_cells_1d = 50
forward_cutoff = 400 # Only make 200 observations (Fourier and pointwise).
my_problem = ToyFourier2d.build_problem(n_cells_1d, forward_cutoff)
updatable_gp = UpdatableGP(kernel, lambda0, sigma0,
            m0, torch.tensor(my_problem.grid.cells).float(), n_chunks=200)

lazy_cov = UpdatableCovLazyTensor(updatable_gp.covariance)

# Test getitem.
lazy_cov[0:10, 0:10].evaluate()

# Test pivoted Cholesky decomposition.
from gpytorch.utils.pivoted_cholesky import pivoted_cholesky

res = pivoted_cholesky(lazy_cov, max_iter=300, error_tol=0.01)
preconditioner = MatmulLazyTensor(res, res.t())

# Now test conjugate gradient inversion.
rhs = torch.rand((lazy_cov.n, 1))
ans = linear_cg(lazy_cov.matmul, rhs, tolerance=0.1, max_iter=400, preconditioner=preconditioner.matmul)
