import os
import torch
import numpy as np
import pandas as pd
from volcapy.grid.grid_from_dsm import Grid
from volcapy.plotting.plot_helper_paper import plot_excu_profile_with_visited_inds_revised
from volcapy.plotting.plot_helper_paper import get_coast_coordinates


data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"
base_results_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/AISTATS_results_v2/FINAL/wIVR_final_small/"
ground_truths_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/AISTATS_results_v2/FINAL/final_samples_matern32/"


def main():
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = grid.cells
    data_coords_surface = np.load(os.path.join(data_folder,"surface_data_coords.npy"))
    data_coords_niklas = np.load(os.path.join(data_folder,"niklas_data_coords.npy"))
    surface_coords = np.load(os.path.join(data_folder,"surface_data_coords.npy"))

    coast_coords = get_coast_coordinates(grid)

    threshold = 2600.0

    def process_sample(sample_nr):
        ground_truth = torch.from_numpy(
                np.load(os.path.join(ground_truths_folder,
                        "prior_sample_{}.npy".format(sample_nr))))

        visited_inds = np.load(
                os.path.join(
                        os.path.join(base_results_folder,
                                "sample_{}/".format(sample_nr)),
                        "visited_inds.npy")
                )
        visited_coords = surface_coords[visited_inds, :]
        out_file_path = "excu_small_with_visited_inds_sample_{}".format(sample_nr)
        plot_excu_profile_with_visited_inds_revised(volcano_coords, data_coords_niklas,
                visited_coords, coast_coords,
                ground_truth, threshold, out_file_path)

    for sample_nr in range(1, 6):
        process_sample(sample_nr)


if __name__ == "__main__":
    main()
