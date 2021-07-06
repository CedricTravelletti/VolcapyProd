import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns


f = h5py.File("/home/cedric/PHD/Dev/NiklasTranslation/original_matlab/Cedric.mat")


# Variable:

# GPS_ground (1, 543) height (in meters) of the data points.
# d_land: (1, 543) observed values.
# Base station is i=26 (Python index).
# dxi: (196, 1): length of each cell (x-axis).
# dyi: (192, 1)
# h_true (1, 543) true height of the data points.
# ind (3, n_cells) index (in the 1D coordinate arrays) of each cell. Note that
# the order iy z, y, x.
# lat (1, 543) latitude of the data points.
# long (1, 543) longitude of the data points.
# sepz: (29, 1) vertical size of each cell.

# Note about the z levels:
#  There are 30 levels (so 29 cells). 
# The z cell number i has top a level i+1 and bottom at level i.
# Hence the top ones have ceiling at 925.
# The lowest one has floor at zbase = -5000.

# So in order to have the correct centroids, one should replace the first value
# of zi with (-475 - (-5000)) / 2.

volcano_coords_x = np.array(f.get('xi')).flatten()
volcano_coords_y = np.array(f.get('yi')).flatten()
volcano_coords_z = np.array(f.get('zi')).flatten()

# First (lowest) cell goes from -475 to -5000.
volcano_coords_z[0] = (-5000 - (-475)) / 2

inds = np.array(f.get('ind')).astype(int) - 1
n_cells = inds.shape[1]

volcano_coords = np.vstack(
        [volcano_coords_x[inds[2, :]],
        volcano_coords_y[inds[1, :]],
        volcano_coords_z[inds[0, :]]]).T.astype(np.float32)

F = np.array(f.get('F_land/data')).reshape((-1, n_cells),
        order='F').astype(np.float32)
d_obs = np.array(f.get('d_land')).flatten().astype(np.float32)

data_x = np.array(f.get('long')).flatten().astype(np.float32)
data_y = np.array(f.get('lat')).flatten().astype(np.float32)
data_z = np.array(f.get('h_true')).flatten().astype(np.float32)

data_coords = np.vstack([data_x, data_y, data_z]).T

# Have to remove first response (base station).
data_coords = data_coords[1:, :]
d_obs = d_obs[1:]

np.save("volcano_coords.npy", np.ascontiguousarray(volcano_coords))
np.save("data_coords.npy", np.ascontiguousarray(data_coords))
np.save("data_obs.npy", np.ascontiguousarray(d_obs))
np.save("F_niklas.npy", np.ascontiguousarray(F))
