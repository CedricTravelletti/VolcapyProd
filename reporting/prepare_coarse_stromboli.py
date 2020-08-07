""" Prepares grid and forward for a downsampled Stromboli.
Needs coarse dsm to begin with.

"""
import numpy as np
from volcapy.grid.grid_from_dsm import Grid
from volcapy.forward import compute_forward
from scipy.spatial import KDTree


dsm_x = np.load("/home/cedric/PHD/Dev/VolcapySIAM/data/dsm_stromboli_x_coarse.npy")
dsm_y = np.load("/home/cedric/PHD/Dev/VolcapySIAM/data/dsm_stromboli_y_coarse.npy")
dsm_z = np.load("/home/cedric/PHD/Dev/VolcapySIAM/data/dsm_stromboli_z_coarse.npy")

z_step = 140
my_grid = Grid.build_grid(dsm_x, dsm_y, dsm_z, z_low=-1600, z_step=200)

print("Grid with {} cells.".format(my_grid.shape[0]))

# Put measurement stations on the whole surface, at 1 meter from the ground.
data_coords = my_grid.surface
data_coords[:, 2] = data_coords[:, 2] + 1.0

# Compute forward on whole surface.
F_full_surface = compute_forward(my_grid.cells, my_grid.cells_roof, my_grid.res_x,
        my_grid.res_y, my_grid.res_z, data_coords, n_procs=4)

# Also compute forward for Niklas data.
from volcapy.loading import load_niklas
niklas_data = load_niklas("/home/cedric/PHD/Dev/Volcano/data/Cedric.mat")

# TODO: WARNING!!! There is one too many data site. Here remove the first, but
# maybe its the last who should be removed.
ref_coords = np.array(niklas_data["data_coords"][0])[None, :]
niklas_data_coords = np.array(niklas_data["data_coords"])[1:]

# TODO: Verify this. According to Niklas, we should subtract the response on
# the reference station. Assuming this is the first data site, then, from every
# line, we should subract the first line.
F_niklas = compute_forward(my_grid.cells, my_grid.cells_roof, my_grid.res_x,
        my_grid.res_y, my_grid.res_z, niklas_data_coords, n_procs=4)

# Subtract the first station.
F_ref_station = compute_forward(my_grid.cells, my_grid.cells_roof, my_grid.res_x,
        my_grid.res_y, my_grid.res_z, ref_coords, n_procs=4)

F_niklas_corr = F_niklas - F_ref_station

# Save everything.
np.save("./input_data/surface_data_coords.npy", data_coords)
np.save("./input_data/niklas_data_coords.npy", niklas_data_coords)
np.save("./input_data/niklas_data_obs.npy", niklas_data['d'])
np.save("./input_data/F_niklas.npy", F_niklas)
np.save("./input_data/F_niklas_corr.npy", F_niklas_corr)
np.save("./input_data/F_full_surface.npy", F_full_surface)
my_grid.save("grid.pickle")
