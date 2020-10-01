""" 
This module provides a class encapsulating all data needed to define an inverse
problem.

The goal is to allow the user to ignore the underlying details of the input
data.

Indeed, if one want to work on Niklas's data, and directly load all 
data from the raw .mat file, one just has to run:

:code:`InverseProblem.from_matfile(path)`


"""
from volcapy import loading
import volcapy.grid.covariance_tools as cl

import numpy as np
import os
from math import floor

from timeit import default_timer as timer


# Unused, just there to remind of the value for Niklas's data.
sigma_d = 0.1

# LOAD DATA
data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"


class InverseProblem():
    """ Numerical implementation of an inverse problem.
    Needs a forward and an inversion grid as
    input. Then can perform several inversions.

    Attributes
    ----------
    cells_coords: 2D array
        Coordinates of the inversion cells. Element i,j should contain j-th
        coordinate of inversion cell nr i. Note the order of the cells should
        reflect the order of the columns of the forward.
    forward: 2D array
        Forward operator.
    data_points
    data_values
    """
    def __init__(self, cells_coords, forward, data_points, data_values):
        self.cells_coords = cells_coords
        self.forward = forward
        self.data_points = data_points
        self.data_values = data_values

        # Dimensions
        self.n_model = cells_coords.shape[0]
        self.n_data = forward.shape[0]

    @classmethod
    def from_matfile(cls, path):
        """ Read forward, inversion grid and data from a matlab file.

        Parameters
        ----------
        path: string
            Path to a matlab file containing the data. The file should have the
            same format as the original file from Niklas (see documentation).

        """
        data = loading.load_niklas(path)

        cells_coords = data['coords']

        # TODO: Maybe refactor loading so that directly returns an array
        # instead of a list.
        # Long run, should think about unifying datatypes.
        data_coords = np.array(data['data_coords'])

        forward = data['F']
        data_values = data['d']

        return cls(cells_coords, forward, data_coords, data_values)

    def subset(self, n_cells):
        """ Subset an inverse problem by only keeping n_cells
        inversion cells selected at random.
        This is used to make the problem more tractable and
        allow prototyping calculations on a small computer.

        The effect of this method is to directly modify the attributes of the
        class to adapt them to the smaller problem.

        Parameters
        ----------
        n_cells: int
            Only keep the first n_cells.

        """
        # Pick indices at random.
        inds = np.random.random_integers(0, self.n_model - 1, size=(n_cells))

        self.cells_coords = self.cells_coords[inds, :]
        self.forward = self.forward[:, inds]
        # We subsetted columns, hence we have to restore C-contiguity by hand.
        self.forward = np.ascontiguousarray(self.forward)
        self.cells_coords = np.asfortranarray(self.cells_coords)

        # Dimensions
        self.n_model = self.cells_coords.shape[0]

    def subset_data(self, n_keep, seed=None):
        """ Subset an inverse problem by only keeping n_keep data
        points selected at random.

        The effect of this method is to directly modify the attributes of the
        class to adapt them to the smaller problem.

        Parameters
        ----------
        n_keep: int
            Only keep n_keep data points.
        seed: int
            Can be used to make choice of test data predictable.

        Returns
        -------
        (rest_forward, rest_data)


        """
        # Pick indices at random.
        if seed is not None:
            np.random.seed(seed)
        inds = np.random.choice(range(self.n_data), n_keep, replace=False)

        # Return the unused part of the forward and of the data so that it can
        # be used as test set.
        rest_forward = np.delete(self.forward, inds, axis=0)
        rest_data = np.delete(self.data_values, inds, axis=0)
        rest_data_points = np.delete(self.data_points, inds, axis=0)

        # Now subset
        self.forward = self.forward[inds, :]
        self.data_points = self.data_points[inds, :]
        self.data_values = self.data_values[inds]
        # We subsetted columns, hence we have to restore C-contiguity by hand.
        self.forward = np.ascontiguousarray(self.forward)

        # New dimensions
        self.n_data = self.forward.shape[0]

        # Note that we return data in a column vector, to make it
        # compatible with the rest of the data.
        return (rest_forward, rest_data[:, None])

    def build_partial_covariance(self, row_begin, row_end, sigma0, lambda0):
        """ (DEPRECATED) Prepare a function for returning partial rows of the covariance
        matrix.

        Warning: should cast, since returns MemoryView.
        """
        n_rows = row_end - row_begin + 1
        print(row_begin)

        dist_mesh = cl.compute_mesh_squared_euclidean_distance(
                self.cells_coords[row_begin:row_end+1, 0],
                self.cells_coords[:, 0],
                self.cells_coords[row_begin:row_end+1, 1],
                self.cells_coords[:, 1],
                self.cells_coords[row_begin:row_end+1, 2],
                self.cells_coords[:, 2])
        dist_mesh = sigma0**2 * np.exp(- 1 / (2 * lambda0**2) * dist_mesh)
        return dist_mesh

    # TODO: Refactor. Effectively, this is chunked multiplication of a matrix with
    # an implicitly defined one.
    def compute_covariance_pushforward(self, G, sigma0, lambda0):
        """ (DEPRECATED) Compute the matrix product C_m * G^T, which we call the *covariance
        pushforward*.

        Parameters
        ----------
        G: 2-D array
            The forward operator.
        """
        # Allocate memory for output array.
        n_data = G.shape[0]
        out = np.zeros((self.n_model, n_data), dtype=np.float32)

        # Store the transpose once and for all.
        GT = (G.T).astype(
                dtype=np.float32, order='C', copy=False)

        # Create the list of chunks.
        chunk_size = 2048
        chunks = mat.chunk_range(self.n_model, chunk_size)

        # Loop in chunks.
        for row_begin, row_end in chunks:
            print(row_begin)
            start = timer()
            # Get corresponding part of covariance matrix.
            partial_cov = self.build_partial_covariance(row_begin, row_end,
                    sigma0, lambda0)

            mid = timer()

            # Append to result.
            out[row_begin:row_end + 1, :] = partial_cov @ GT

            end = timer()
            print(
                "Building done in "
                + str(mid - start)
                + " and multiplication done in " + str(end - mid))

        return out

    # TODO: Factor out some methods.
    def inverse(self, out_folder, prior_mean, sigma_d,
            sigma0, lambda0,
            preload_covariance_pushforward=False, cov_pushforward=None,
            compute_post_covariance=False):
        """ (DEPRECATED) Perform inversion.

        Parameters
        ----------
        out_folder: string
            Path to a folder where we will save the resutls.
        compute_post_covariance: Boolean
            If set to True, then will compute and store the full posterior
            covariance matrix (huge).

        """
        # We will save a lot of stuff, so place it in a dedicated directory..
        os.chdir(out_folder)

        # Build prior mean.
        # The order parameters are statically compiled in fast covariance.
        m_prior = np.full(self.n_model, prior_mean, dtype=np.float32)

        # Build data covariance matrix.
        cov_d = sigma_d**2 * np.eye(self.n_data, dtype=np.float32)

        # If the covariance pushforward hasnt been precomputed.
        if not preload_covariance_pushforward:
            # Compute big matrix product and save.
            print("Computing big matrix product.")
            start = timer()
            self.covariance_pushforward = self.compute_covariance_pushforward(
                    self.forward, sigma0, lambda0)
            end = timer()
            print("Done in " + str(end - start))
            np.save('Cm_Gt.npy', self.covariance_pushforward)
        else:
            self.covariance_pushforward = cov_pushforward

        # Use to perform inversion and save.
        print("Inverting matrix.")
        start = timer()
        temp = self.forward @ self.covariance_pushforward
        inverse = np.linalg.inv(temp + cov_d)
        end = timer()
        print("Done in " + str(end - start))

        print("Computing posterior mean.")
        start = timer()
        m_posterior = (m_prior
                + self.covariance_pushforward
                @ inverse @ (self.data_values - self.forward @ m_prior))
        self.m_posterior = m_posterior

        end = timer()
        print("Done in " + str(end - start))

        np.save('m_posterior.npy', m_posterior)

        # ----------------------------------------------
        # Build and save the diagonal of the posterior covariance matrix.
        # ----------------------------------------------
        print("Computing posterior variance.")
        start = timer()
        Cm_post = np.empty([self.n_model],
                dtype=self.covariance_pushforward.dtype)

        #TODO: Find a name for these and store them cleanly.
        A = self.covariance_pushforward @ inverse
        B = self.covariance_pushforward.T

        # Diagonal is just given by the scalar product of the two factors.
        for i in range(self.n_model):
            Cm_post[i] = np.dot(A[i, :], B[:, i])

        end = timer()
        print("Done in " + str(end - start))

        # Save the square root standard deviation).
        """
        np.save(
                "posterior_cov_diag.npy",
                np.sqrt(np.array([sigma0] * self.n_model) - Cm_post))
        """
        print("DONE")

        if compute_post_covariance:
            self.compute_post_covariance()

    # TODO: Refactor. Currently accessing stuf oustide its scope.
    def compute_post_covariance(self):
        """ (DEPRECATED) Compute the full posterior covariance matrix.
        """
        # AMBITIOUS: Compute the whole (120GB) posterior covariance matrix.
        print("Computing posterior covariance.")
        post_cov = np.memmap('post_cov.npy', dtype='float32', mode='w+',
                shape=(self.n_model, self.n_model))

        # Compute the matrix product line by line.
        # TODO: Could compute several lines at a time, by chunks.
        for i in range(self.n_model):
            print(i)
            prior_cov = self.build_partial_covariance(i, i)
            post_cov[i, :] = prior_cov - A[i, :] @ B

        # Flush to disk.
        del post_cov

    def save_squared_distance_mesh(self, path):
        """ (DEPRECATED) Computes the matrix of squared euclidean distances betwen all
        couples and saves to disk.
        """
        # AMBITIOUS: Compute the whole (120GB) posterior covariance matrix.
        print("Computing posterior covariance.")
        post_cov = np.memmap('post_cov.npy', dtype='float32', mode='w+',
                shape=(self.n_model, self.n_model))

        # Compute the matrix product line by line.
        # TODO: Could compute several lines at a time, by chunks.
        for i in range(self.n_model):
            print(i)
            prior_cov = self.build_partial_covariance(i, i)
            post_cov[i, :] = prior_cov - A[i, :] @ B

        # Flush to disk.
        del post_cov
