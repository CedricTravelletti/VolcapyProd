""" Plots a given ground truth (saved as npy) in 3D.

"""
import os
import torch
import numpy as np
from volcapy.grid.grid_from_dsm import Grid
import matplotlib.pyplot as plt
from volcapy.plotting.plot_3D import point_cloud_to_2D_regular_grid
from volcapy.plotting.plot_3D import underground_point_cloud_to_structured_grid
from mayavi import mlab


data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"


def main():
    # Load
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = grid.cells
    data_coords = np.load(os.path.join(data_folder,"surface_data_coords.npy"))
    ground_truth = np.load("./ground_truth.npy")
    data_values = np.load(os.path.join(data_folder, "post_data_sample.npy"))
    surface_coords = grid.surface

    # Interpolate surface on regular grid.
    X_surface_mesh, Y_surface_mesh, Z_surface_mesh = point_cloud_to_2D_regular_grid(
            surface_coords, n_grid=300)

    density_structured_grid = underground_point_cloud_to_structured_grid(
                    volcano_coords, ground_truth, surface_coords,
                    n_grid=50, fill_offset=1, fill_value=-1e6)

    # Threshold to only plot one region.
    vals[vals > 500] = 0
    vals[vals < 0] = 0

    
    mlab.surf(X_surface_mesh, Y_surface_mesh, Z_surface_mesh, opacity=0.35)
    d = mlab.pipeline.add_dataset(density_structured_grid)
    iso = mlab.pipeline.iso_surface(d)
    mlab.show()


if __name__ == "__main__":
    main()
