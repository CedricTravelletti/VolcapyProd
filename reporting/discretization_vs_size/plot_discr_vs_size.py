""" Discretize the Stromboli at different lenghts and see how the number of
cells (and thus the size of the matrices) vary.

This script was built by fusing parts of the scripts in
volcapy.data_preparation.

"""
import numpy as np
import os
import h5py
from skimage.transform import downscale_local_mean
from scipy.spatial import KDTree
from volcapy.grid.grid_from_dsm import Grid
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()
# sns.set_style("whitegrid", {'axes.grid' : False})

# Trygve's plot parameters.
plt.rcParams["font.family"] = "Times New Roman"
plot_params = {
        'font.size': 20, 'font.style': 'oblique',
        # 'xtick.labelsize': 'x-small',
        'axes.labelsize': 'medium',
        'axes.titlesize':'medium',
        'xtick.major.pad': '1',
        'xtick.minor.pad': '1',
        'ytick.major.pad': '1'}
plt.rcParams.update(plot_params)

input_path = "/home/cedric/PHD/Dev/VolcapySIAM/data/original/Cedric.mat"


def main(nx=50, ny=50):
    dataset = h5py.File(input_path, 'r')

    # Load the arrays.
    dsm_x = np.array(dataset['x'], dtype=np.float32)
    dsm_y = np.array(dataset['y'], dtype=np.float32)
    dsm_z = np.array(dataset['z'], dtype=np.float32)

    # Cut the outer regions.
    print(dsm_x.shape)
    low_x_cutoff_ind = 250
    low_y_cutoff_ind = 243
    high_x_cutoff_ind = -250
    high_y_cutoff_ind = -243

    dsm_x = dsm_x[low_x_cutoff_ind:high_x_cutoff_ind]
    dsm_y = dsm_y[low_y_cutoff_ind:high_y_cutoff_ind]
    dsm_z = dsm_z[low_x_cutoff_ind:high_x_cutoff_ind, low_y_cutoff_ind:high_y_cutoff_ind]

    # Find boundaries.
    x_min = np.min(dsm_x)
    x_max = np.max(dsm_x)
    y_min = np.min(dsm_y)
    y_max = np.max(dsm_y)

    def discretize_in_n(n):
        """ Discretize the geometry at a given lengthscale, 
        here specifiec by the number of cells along each direction of the
        horizontal plane.
    
        """
        nx = n
        ny = n
        # Regrid.
        x, step_x = np.linspace(x_min, x_max, nx, endpoint=True, retstep=True)
        y, step_y = np.linspace(y_min, y_max, ny, endpoint=True, retstep=True)
        print("Step x {}".format(step_x))
        print("Step y {}".format(step_y))
    
        # Set the same resolution for z.
        z_step = float(np.mean([step_x, step_y]))
        print("Step z {}".format(z_step))
    
        # Find z by getting closest cell in fine grid.
        tree_x = KDTree(dsm_x)
        tree_y = KDTree(dsm_y)
        _, inds_x = tree_x.query(x[:, None])
        _, inds_y = tree_y.query(y[:, None])
    
        # Remesh the z-s.
        a, b = np.meshgrid(inds_x, inds_y)
        z = dsm_z[a, b]
        print("Minimal altitude {}".format(np.min(z)))
    
        my_grid = Grid.build_grid(x, y, z, z_low=-1600, z_step=z_step)
        print("Grid with {} cells.".format(my_grid.shape[0]))
        print(my_grid.res_x)
        print(my_grid.res_y)
        print(my_grid.res_z)

        # Return mean resolution and total number of cells.
        return my_grid.shape[0], np.mean(np.array([step_x, step_y, z_step]))

    ns_cells, ress = [], []
    for n in np.arange(30, 280, step=8):
        n_cells, mean_res = discretize_in_n(n)
        ns_cells.append(n_cells)
        ress.append(mean_res)
    
    fig, ax = plt.subplots()
    ax.semilogy(ress, ns_cells, label="discretization grid", color="blue")
    # Invert x axis.
    ax.set_xlim(max(ress),min(ress))
    ax.set_ylabel("Number of cells")
    ax.set_xlabel("Resolution [m]")

    # Plot memory footprint.
    mem_cov = 1e-9 * 4 * np.multiply(ns_cells, ns_cells)
    n_data = 500
    mem_pushfwd = 1e-9 * 4 * 500 * np.array(ns_cells)

    ax2 = ax.twinx()
    ax2.semilogy(ress, mem_cov, color="red", label="covariance matrix")
    ax2.semilogy(ress, mem_pushfwd, color="green", label="covariance pushforward")
    ax2.set_ylabel("Memory [GB]")

    ax2.axhline(16, ls='--', color="black")
    ax2.text(45,17, "16 GB", color="black", size=14)

    fig.legend(loc="upper right", bbox_to_anchor=(1,1),
            bbox_transform=ax.transAxes)

    plt.savefig("discretization_vs_size.png", bbox_inches='tight', pad_inches=0, dpi=400)
    plt.show()

    print(ress)

if __name__ == "__main__":
    main()
