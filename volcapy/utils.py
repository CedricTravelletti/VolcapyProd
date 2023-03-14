""" Various utility functions.

"""
import numpy as np
import torch
from sklearn.cluster import KMeans


def _make_column_vector(y):
    """ Make sure the data is a column vector.

    """
    if ((len(y.shape) >= 2) and (y.shape[1] == 1)):
        return y
    elif len(y.shape) == 1:
        return y.reshape(-1, 1)
    else:
        raise ValueError("Shape of data vector {} is not valid. Please provide a column vector.".format(y.shape))

# Generate R-sequences in arbitrary dimensions.
# Those are space filling sequences.
# Based on
# https://stats.stackexchange.com/questions/25528/do-low-discrepancy-sequences-work-in-discrete-spaces
# and
# http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/  


# Use Newton-Rhapson-Method to compute the plastic constant.
def gamma(d):
    x=1.0000
    for i in range(20):
        x = x-(pow(x,d+1)-x-1)/((d+1)*pow(x,d)-1)
    return x


def r_sequence(n_samples, dim, n_cells_1d):
    """ Generate a space filling sequence in a discretized hypercube.

    Parameters
    ----------
    n_samples: int
        How many samples to generate.
    dim: int
        Dimension of the hypercube.
    n_cells_1d: int
        Side lenghts of the hypercube.

    Returns
    -------
    samples: (n_samples, dim) array

    """
    g = gamma(dim)
    alpha = np.zeros(dim)                 
    for j in range(dim):
        alpha[j] = pow(1/g,j+1) %1
    z = np.zeros((n_samples, dim))    
    c = np.zeros((n_samples,dim)).astype(int)  
    
    for i in range(n_samples):
        z[i, :] = (0.5 + alpha*(i+1)) %1
        c[i, :] = (np.floor(n_cells_1d *z[i, :])).astype(int)
    return c

def kMeans_clustering(k, data_coords, init='k-means++', n_init=10, random_state=42):
    """ Cluster data points into k clusters using kMeans.

    Parameters
    ----------
    k: int
        Number of clusters.
    data_coords: array (n_data, n_dims)
        Coordinates of the points to cluster.
    init: 'random' or 'k-means++', default is 'k-means++'
        Specifies how to initialize the cluster centroids.
        Use random to get different clusters.
    Returns
    -------
    cluster_labels: array (n_data)
        For each data point, the lable (int) of the cluster 
        it belongs to. Labels start at 0.

    """
    if torch.is_tensor(data_coords): data_coords = data_coords.cpu().numpy()
    kmeans = KMeans(init=init, n_clusters=k,
            n_init=n_init, max_iter=300, random_state=random_state,)
    kmeans.fit(data_coords)
    return kmeans.labels_

def kMeans_folds(k, data_coords, init='k-means++', n_init=10, random_state=42):
    """ Computes folds using kMeans clustering. 
    Returns a list of folds, each fold being described by a list of 
    indices, corresponding to the points that belong to the cluster. 
    One can then use each of theses lists as a list of points to be 
    left out in cross-validation. 

    Parameters
    ----------
    k: int
        Number of clusters.
    data_coords: array (n_data, n_dims)
        Coordinates of the points to cluster.

    Returns
    -------
    folds: List[array]
        List of integer arrays, each describing the indices 
        belonging to a given fold.

    """
    cluster_labels = kMeans_clustering(k, data_coords,
            init=init, n_init=n_init, random_state=random_state)
    folds = []
    for i in range(k):
        fold_indices = np.argwhere(cluster_labels == i).flatten()
        folds.append(fold_indices)
    return folds
