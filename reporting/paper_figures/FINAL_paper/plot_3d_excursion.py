import os
import torch
import numpy as np
import pandas as pd
from volcapy.grid.grid_from_dsm import Grid
from volcapy.plotting.plot_helper_paper import plot_excu_profile_with_data_revised
from volcapy.plotting.plot_helper_paper import get_coast_coordinates


data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"
base_results_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/AISTATS_results_v2/FINAL/"
ground_truths_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/AISTATS_results_v2/FINAL/final_samples_matern32/"


def main():
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = grid.cells
    data_coords_surface = np.load(os.path.join(data_folder,"surface_data_coords.npy"))
    data_coords_niklas = np.load(os.path.join(data_folder,"niklas_data_coords.npy"))

    coast_coords = get_coast_coordinates(grid)

    threshold_small = 2600.0
    threshold_big = 2500.0

    for i in range(1, 6):
        ground_truth = torch.from_numpy(
            np.load(os.path.join(ground_truths_folder,
                    "prior_sample_{}.npy".format(i))))

        out_file_path = "excu_profile_small_{}".format(i)
        plot_excu_profile_with_data_revised(volcano_coords, data_coords_niklas,
            coast_coords,
            ground_truth, threshold_small, out_file_path)

        out_file_path = "excu_profile_big_{}".format(i)
        plot_excu_profile_with_data_revised(volcano_coords, data_coords_niklas,
            coast_coords,
            ground_truth, threshold_big, out_file_path)



if __name__ == "__main__":
    main()
