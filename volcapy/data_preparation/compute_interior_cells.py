""" Determines which cells lie deep inside the volcano (i.e. away from the
surface).
This is useful when using IVR to select new datapoints, since we are mostly
interested at variance reduction deep below the surface.

"""
import numpy as np
import os
from volcapy.grid.grid_from_dsm import Grid
from volcapy.forward import compute_forward
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud, _array_to_point_cloud
from scipy.spatial import KDTree


DEPTH_THRESHOLD = 150.0 # Only keep cells at least that far from the surface.

output_path = "/home/cedric/PHD/Dev/VolcapySIAM/data/inversion_data_dsm_coarse"

dsm_x = np.load("/home/cedric/PHD/Dev/VolcapySIAM/data/dsm_coarse/dsm_stromboli_x_coarse.npy")
dsm_y = np.load("/home/cedric/PHD/Dev/VolcapySIAM/data/dsm_coarse/dsm_stromboli_y_coarse.npy")
dsm_z = np.load("/home/cedric/PHD/Dev/VolcapySIAM/data/dsm_coarse/dsm_stromboli_z_coarse.npy")

my_grid = Grid.build_grid(dsm_x, dsm_y, dsm_z, z_low=-1600, z_step=50)

print("Grid with {} cells.".format(my_grid.shape[0]))

# Get surface and put in KDTree for easy nearest one computation.
surface_cells = my_grid.surface
from scipy.spatial import KDTree
surface_tree = KDTree(surface_cells)

# Loop over cells and only keep the ones that are far away from the surface.
deep_cells_inds = []
for i, cell in enumerate(my_grid.cells):
    print("Processing cell {}/{}".format(i, my_grid.cells.shape[0]))
    d, _ = surface_tree.query(cell)
    
    if d > DEPTH_THRESHOLD:
        deep_cells_inds.append(i)

print("There are {} surface cells.".format(surface_cells.shape[0]))
print("Kept {}/{} cells".format(len(deep_cells_inds), my_grid.cells.shape[0]))

np.save(os.path.join(output_path, "deep_cells_inds.npy"),
        np.array(deep_cells_inds, dtype=np.int32))
_array_to_point_cloud(my_grid.cells[deep_cells_inds, :],
        np.ones(len(deep_cells_inds)),
        os.path.join(output_path, "deep_cells.vtk"))
irregular_array_to_point_cloud(my_grid.cells[my_grid.surface_inds],
        np.ones(len(my_grid.surface_inds)),
        os.path.join(output_path, "surface_cells.vtk"), fill_nan_val=-20000.0)
