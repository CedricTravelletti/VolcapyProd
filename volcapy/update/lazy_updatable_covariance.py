""" Wrap our lazy covariance approach into a GPytorch lazy tensor.

"""
import torch
from gpytorch.lazy import LazyTensor, MatmulLazyTensor, DiagLazyTensor
from gpytorch.utils.broadcasting import _matmul_broadcast_shape, _mul_broadcast_shape
import gpytorch.settings as settings


class UpdatableCovLazyTensor(LazyTensor):
    def __init__(self, updatable_cov):
        super(UpdatableCovLazyTensor, self).__init__(updatable_cov.cells_coords)
        self.updatable_cov = updatable_cov
        self.n = updatable_cov.n_cells

    def _matmul(self, rhs):
        return self.updatable_cov.mul_right(rhs).float()

    def _transpose_nonbatch(self):
        # Covariances are symmetric matrices
        return self

    def _size(self):
        return torch.Size([self.n, self.n])

    def _get_indices(self, row_index, col_index, *batch_indices):
        result = torch.zeros(row_index.shape[0])
        for count, (i, j) in enumerate(zip(row_index, col_index)):
            # Creat basis vector that extract corresponding column.
            e_j = torch.zeros((self.n, 1))
            e_j[j] = 1.0
            result[count] = self._matmul(e_j)[i]
        return result

    def _approx_diag(self):
        """ Extracts the diagonal of the UpdatableCovariance.

        """
        return self.updatable_cov.extract_variance()

    def _getitem(self, row_index, col_index, *batch_indices):
        col_indexer = DiagLazyTensor(torch.ones(self.n))[:, col_index]
        row_indexer = DiagLazyTensor(torch.ones(self.n))[:, row_index]
        res = MatmulLazyTensor(MatmulLazyTensor(self, row_indexer).t(), col_indexer)
        return res

    def representation(self):
        return [self.updatable_cov]
