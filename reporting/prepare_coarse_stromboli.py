""" Prepares grid and forward for a downsampled Stromboli.

"""
import numpy as np
from volcapy.grid.grid_from_dsm import Grid
from volcapy.forward import compute_forward
from scipy.spatial import KDTree


dsm_x = np.load("/home/cedric/PHD/Dev/VolcapySIAM/data/dsm_stromboli_x_coarse.npy")
dsm_y = np.load("/home/cedric/PHD/Dev/VolcapySIAM/data/dsm_stromboli_y_coarse.npy")
dsm_z = np.load("/home/cedric/PHD/Dev/VolcapySIAM/data/dsm_stromboli_z_coarse.npy")

my_grid = Grid.build_grid(dsm_x, dsm_y, dsm_z, z_low=-450, z_step=120)

# Put measurement stations on the whole surface, at 1 meter from the ground.
data_coords = my_grid.surface
data_coords[:, 2] = data_coords[:, 2] + 1.0

F = compute_forward(my_grid.cells, my_grid.cells_roof, my_grid.res_x,
        my_grid.res_y, my_grid.res_z, data_coords, n_procs=4)

# Also compute forward for Niklas data.
from volcapy.loading import load_niklas
niklas_data = load_niklas("/home/cedric/PHD/Dev/Volcano/data/Cedric.mat")
niklas_data_coords = np.array(niklas_data["data_coords"])

F_niklas = compute_forward(my_grid.cells, my_grid.cells_roof, my_grid.res_x,
        my_grid.res_y, my_grid.res_z, niklas_data_coords, n_procs=4)

# Save everything.
np.save("surface_data_coords.npy", data_coords)
np.save("niklas_data_coords.npy", niklas_data_coords)
np.save("F_niklas.npy", F_niklas)
np.save("F.npy", F)
my_grid.save("grid.pickle")
