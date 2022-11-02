""" Prepares grid and forward for a downsampled Stromboli.
Needs coarse dsm to begin with.

"""
import numpy as np
import os
import sys
from volcapy.grid.grid_from_dsm import Grid
from volcapy.forward.gravimetry import compute_forward
from scipy.spatial import KDTree


ORIGINAL_NIKLAS_DATA_PATH =  "/home/cedric/PHD/Dev/VolcapySIAM/data/original/Cedric.mat"   
DEFAULT_OUTPATH = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas"

def prepare_stromboli(dsm_path, niklas_data_path, output_path, z_step):
    dsm_x = np.load(os.path.join(dsm_path, "dsm_stromboli_x.npy"))
    dsm_y = np.load(os.path.join(dsm_path, "dsm_stromboli_y.npy"))
    dsm_z = np.load(os.path.join(dsm_path, "dsm_stromboli_z.npy"))
    
    my_grid = Grid.build_grid(dsm_x, dsm_y, dsm_z, z_low=-800, z_step=z_step)
    print("Grid with {} cells.".format(my_grid.shape[0]))

    # Name the output directory with the number of cells.
    dir_name = "stromboli_{}_cells".format(my_grid.shape[0])
    output_path = os.path.join(output_path, dir_name)
    os.makedirs(output_path, exist_ok=True)
    
    my_grid.save(os.path.join(output_path, "grid.pickle"))

    # Determines which cells lie deep inside the volcano (i.e. away from the
    # surface).
    # This is useful when using IVR to select new datapoints, since we are mostly
    # interested at variance reduction deep below the surface.
    DEPTH_THRESHOLD = 150.0 # Only keep cells at least that far from the surface.
    surface_tree = KDTree(my_grid.surface)

    # Loop over cells and only keep the ones that are far away from the surface.
    deep_cells_inds = []
    for i, cell in enumerate(my_grid.cells):
        print("Processing cell {}/{}".format(i, my_grid.cells.shape[0]))
        d, _ = surface_tree.query(cell)
        if d > DEPTH_THRESHOLD:
            deep_cells_inds.append(i)
    
    print("There are {} surface cells.".format(my_grid.surface.shape[0]))
    print("Kept {}/{} cells".format(len(deep_cells_inds), my_grid.cells.shape[0]))
    
    np.save(os.path.join(output_path, "deep_cells_inds.npy"),
            np.array(deep_cells_inds, dtype=np.int32))
    
    # Put measurement stations on the whole surface, at 1 meter from the ground.
    fine_surface_alts = my_grid.cells_roof[my_grid.surface_inds] # Get altitudes from the fine DSM.
    fine_surface = my_grid.surface
    fine_surface[:, 2] = fine_surface_alts

    # First only keep cells that are outside the water.
    data_coords = np.array([x for x in fine_surface if x[2] > 0.0])
    print(("There are {} surface cells above sea level. "
            + "Placing potential observation "
            + "sites on each of those..").format(data_coords.shape[0]))
    data_coords[:, 2] = data_coords[:, 2] + 1.0

    # Also load locations of Niklas data points.
    from volcapy.loading import load_niklas
    niklas_data = load_niklas(niklas_data_path)
    niklas_data_coords = np.array(niklas_data["data_coords"])[1:]
    niklas_data_values = np.array(niklas_data["d"])

    # Find the indices of the closest data points to Niklas data.
    # This may be used to mitigate the difference between the two in case there
    # are a lot of difference in altitude because of the discretization.
    tree_surf_data = KDTree(data_coords)
    _, inds_niklas_in_surface = tree_surf_data.query(niklas_data_coords)
    niklas_data_coords_insurf = data_coords[inds_niklas_in_surface]
    np.save(os.path.join(output_path, "niklas_data_inds_insurf.npy"),
            inds_niklas_in_surface)

    # Save location of data points before computing forward.
    # Save the surface of the volcano and the datapoints. in vtk.
    from volcapy.plotting.vtkutils import irregular_array_to_point_cloud, array_to_vector_cloud
    irregular_array_to_point_cloud(my_grid.surface,
            np.ones(my_grid.surface.shape[0]),
            os.path.join(output_path, "surface.vtk"), fill_nan_val=-20000.0)
    
    orientation_data = np.zeros((data_coords.shape[0], 3))
    orientation_data[:, 2] = -1.0
    array_to_vector_cloud(data_coords,
            orientation_data,
            os.path.join(output_path, "data_points.vtk"))

    orientation_data = np.zeros((niklas_data_coords.shape[0], 3))
    orientation_data[:, 2] = -1.0
    orientation_data_vals = np.zeros((niklas_data_coords.shape[0], 3))
    orientation_data_vals[:, 2] = niklas_data_values
    array_to_vector_cloud(niklas_data_coords,
            orientation_data,
            os.path.join(output_path, "niklas_data_coords.vtk"))
    array_to_vector_cloud(niklas_data_coords_insurf,
            orientation_data,
            os.path.join(output_path, "niklas_data_coords_insurf.vtk"))
    array_to_vector_cloud(niklas_data_coords,
            orientation_data_vals,
            os.path.join(output_path, "niklas_data_vals.vtk"))

    # ----------------------------------------------------------------------
    
    # Also compute forward for Niklas data.
    # TODO: WARNING!!! There is one too many data site. Here remove the first, but
    # maybe its the last who should be removed.
    ref_coords = np.array(niklas_data["data_coords"][0])[None, :]
    
    # TODO: Verify this. According to Niklas, we should subtract the response on
    # the reference station. Assuming this is the first data site, then, from every
    # line, we should subract the first line.
    F_ref_station = compute_forward(my_grid.cells, my_grid.cells_roof, my_grid.res_x,
            my_grid.res_y, my_grid.res_z, ref_coords, n_procs=4)

    F_niklas = compute_forward(my_grid.cells, my_grid.cells_roof, my_grid.res_x,
            my_grid.res_y, my_grid.res_z, niklas_data_coords, n_procs=4)
    
    # Subtract the first station.
    F_niklas_corr = F_niklas - np.repeat(F_ref_station, F_niklas.shape[0],
            axis=0)

    np.save(os.path.join(output_path, "F_niklas.npy"), F_niklas)
    np.save(os.path.join(output_path, "F_niklas_corr.npy"), F_niklas_corr[1:])
    np.save(os.path.join(output_path, "niklas_data_coords.npy"),
            niklas_data_coords[1:, :])
    np.save(os.path.join(output_path, "niklas_data_obs.npy"),
            niklas_data['d'][1:])

    
    # Compute forward on whole surface.
    F_full_surface = compute_forward(my_grid.cells, my_grid.cells_roof, my_grid.res_x,
            my_grid.res_y, my_grid.res_z, data_coords, n_procs=4)
    
    # Save everything, but cut first observation.
    np.save(os.path.join(output_path, "surface_data_coords.npy"), data_coords)
    np.save(os.path.join(output_path, "F_full_surface.npy"), F_full_surface)
    

if __name__ == "__main__":
    print("Usage: prepare_stromboli.py dsm_path niklas_data_path, output_path z_step")
    dsm_path, niklas_data_path, output_path, z_step = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    prepare_stromboli(dsm_path, niklas_data_path, output_path, z_step=z_step)
