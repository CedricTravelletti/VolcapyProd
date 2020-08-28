""" Downsample the Stromboli DSM to a more manageable size.

"""
import numpy as np
import os
import sys
import h5py
from skimage.transform import downscale_local_mean
from scipy.spatial import KDTree


ORIGINAL_NIKLAS_DATA_PATH = "/home/cedric/PHD/Dev/VolcapySIAM/data/original/Cedric.mat"
DEFAULT_OUTPATH = "/home/cedric/PHD/Dev/VolcapySIAM/data/DSMs"

NX, NY = 120, 120


def downsample_stromboli(output_path, nx=NX, ny=NY):
    dataset = h5py.File(ORIGINAL_NIKLAS_DATA_PATH, 'r')

    # Load the arrays.
    dsm_x = np.array(dataset['x_fine'], dtype=np.float32)
    dsm_y = np.array(dataset['y_fine'], dtype=np.float32).reshape(-1, 1)
    dsm_z = np.array(dataset['z_fine'], dtype=np.float32)

    """
    # Cut the outer regions.
    low_x_cutoff_ind = 250
    low_y_cutoff_ind = 243
    # high_x_cutoff_ind = -250
    high_x_cutoff_ind = -310
    # high_y_cutoff_ind = -243
    high_y_cutoff_ind = -303

    dsm_x = dsm_x[low_x_cutoff_ind:high_x_cutoff_ind]
    dsm_y = dsm_y[low_y_cutoff_ind:high_y_cutoff_ind]
    dsm_z = dsm_z[low_x_cutoff_ind:high_x_cutoff_ind, low_y_cutoff_ind:high_y_cutoff_ind]
    """

    # Find boundaries.
    x_min = np.min(dsm_x)
    x_max = np.max(dsm_x)
    y_min = np.min(dsm_y)
    y_max = np.max(dsm_y)

    # Regrid.
    x, x_step = np.linspace(x_min, x_max, nx, retstep=True, endpoint=True)
    y, y_step = np.linspace(y_min, y_max, ny, retstep=True, endpoint=True)

    print("x:y resolution {}:{}".format(x_step, y_step))

    # Find z by getting closest cell in fine grid.
    tree_x = KDTree(dsm_x)
    tree_y = KDTree(dsm_y)
    _, inds_x = tree_x.query(x[:, None])
    _, inds_y = tree_y.query(y[:, None])

    # Remesh the z-s.
    a, b = np.meshgrid(inds_x, inds_y, indexing='ij')
    z = dsm_z[a, b]


    # Name the output directory with the resolution.
    dir_name = "dsm_res_x{}_y{}".format(int(x_step), int(y_step))
    output_path = os.path.join(output_path, dir_name)
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, "dsm_stromboli_x.npy"), x)
    np.save(os.path.join(output_path, "dsm_stromboli_y.npy"), y)
    np.save(os.path.join(output_path, "dsm_stromboli_z.npy"), z)

if __name__ == "__main__":
    print("Usage: downsample_stromboli NX NY")
    NX, NY = int(sys.argv[1]), int(sys.argv[2])
    downsample_stromboli(output_path=DEFAULT_OUTPATH, nx=NX, ny=NY)
