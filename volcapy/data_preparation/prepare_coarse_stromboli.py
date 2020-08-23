""" Prepares grid and forward for a downsampled Stromboli.
Needs coarse dsm to begin with.

"""
import numpy as np
import os
import sys
from volcapy.grid.grid_from_dsm import Grid
from volcapy.forward import compute_forward
from scipy.spatial import KDTree


ORIGINAL_NIKLAS_DATA_PATH =  "/home/cedric/PHD/Dev/VolcapySIAM/data/original/Cedric.mat"   

# input_path = "/home/cedric/PHD/Dev/VolcapySIAM/data/dsm_coarse"
input_path = "/home/cedric/PHD/Dev/VolcapySIAM/data/dsm_lesscoarse"
# output_path = "/home/cedric/PHD/Dev/VolcapySIAM/data/inversion_data_dsm_lesscoarse"
# output_path = "/home/cedric/PHD/Dev/ODEBO/example_runs/volcano/input_data"
output_path = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/variance_extraction_benchmark"

def prepare_stromboli(input_path, output_path, z_step):
    dsm_x = np.load(os.path.join(input_path, "dsm_stromboli_x.npy"))
    dsm_y = np.load(os.path.join(input_path, "dsm_stromboli_y.npy"))
    dsm_z = np.load(os.path.join(input_path, "dsm_stromboli_z.npy"))
    
    my_grid = Grid.build_grid(dsm_x, dsm_y, dsm_z, z_low=-1500, z_step=z_step)
    
    print("Grid with {} cells.".format(my_grid.shape[0]))
    
    # Name the output directory with the number of cells.
    dir_name = "stromboli_{}_cells".format(my_grid.shape[0])
    output_path = os.path.join(output_path, dir_name)
    os.mkdir(output_path)
    
    # Put measurement stations on the whole surface, at 1 meter from the ground.
    data_coords = my_grid.surface
    data_coords[:, 2] = data_coords[:, 2] + 1.0
    
    # Compute forward on whole surface.
    F_full_surface = compute_forward(my_grid.cells, my_grid.cells_roof, my_grid.res_x,
            my_grid.res_y, my_grid.res_z, data_coords, n_procs=4)
    
    # Also compute forward for Niklas data.
    from volcapy.loading import load_niklas
    niklas_data = load_niklas(ORIGINAL_NIKLAS_DATA_PATH)
    
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
    np.save(os.path.join(output_path, "surface_data_coords.npy"), data_coords)
    np.save(os.path.join(output_path, "niklas_data_coords.npy"), niklas_data_coords)
    np.save(os.path.join(output_path, "niklas_data_obs.npy"), niklas_data['d'])
    np.save(os.path.join(output_path, "F_niklas.npy"), F_niklas)
    np.save(os.path.join(output_path, "F_niklas_corr.npy"), F_niklas_corr)
    np.save(os.path.join(output_path, "F_full_surface.npy"), F_full_surface)
    my_grid.save(os.path.join(output_path, "grid.pickle"))
    
    # Save the surface of the volcano in vtk.
    from volcapy.plotting.vtkutils import irregular_array_to_point_cloud
    irregular_array_to_point_cloud(my_grid.surface,
            np.ones(my_grid.surface.shape[0]),
            os.path.join(output_path, "surface.vtk"), fill_nan_val=-20000.0)

if __name__ == "__main__":
    print("Usage: prepare_stromboli.py input_path output_path z_step")
    input_path, output_path, z_step = sys.argv[1], sys.argv[2], sys.argv[3]
    prepare_stromboli(input_path, output_path, z_step)
