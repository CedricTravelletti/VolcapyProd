""" Downsample the Stromboli DSM to a more manageable size.

"""
import numpy as np
import os
import h5py
from skimage.transform import downscale_local_mean
from scipy.spatial import KDTree

def main(path, nx=50, ny=50):
    dataset = h5py.File(path, 'r')

    # Load the arrays.
    dsm_x = np.array(dataset['x'], dtype=np.float32)
    dsm_y = np.array(dataset['y'], dtype=np.float32)
    dsm_z = np.array(dataset['z'], dtype=np.float32)

    # Cut the outer regions.
    dsm_x = dsm_x[50:-80]
    dsm_y = dsm_y[50:-80]
    dsm_z = dsm_z[50:-80, 50:-80]

    # Find boundaries.
    x_min = np.min(dsm_x)
    x_max = np.max(dsm_x)
    y_min = np.min(dsm_y)
    y_max = np.max(dsm_y)

    # Regrid.
    x = np.linspace(x_min, x_max, nx, endpoint=True)
    y = np.linspace(y_min, y_max, ny, endpoint=True)

    # Find z by getting closest cell in fine grid.
    tree_x = KDTree(dsm_x)
    tree_y = KDTree(dsm_y)
    _, inds_x = tree_x.query(x[:, None])
    _, inds_y = tree_y.query(y[:, None])

    # Remesh the z-s.
    a, b = np.meshgrid(inds_x, inds_y)
    z = dsm_z[a, b]

    out_path = "/home/cedric/PHD/Dev/VolcapySIAM/data/dsm_coarse"
    np.save(os.path.join(out_path, "dsm_stromboli_x_coarse.npy"), x)
    np.save(os.path.join(out_path, "dsm_stromboli_y_coarse.npy"), y)
    np.save(os.path.join(out_path, "dsm_stromboli_z_coarse.npy"), z)

if __name__ == "__main__":
    main("/home/cedric/PHD/Dev/VolcapySIAM/data/original/Cedric.mat")
