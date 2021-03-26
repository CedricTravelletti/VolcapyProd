import os
import torch
import numpy as np
from volcapy.grid.grid_from_dsm import Grid
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud, _array_to_point_cloud
from volcapy.plotting.vtkutils import array_to_vector_cloud


data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"


def main(sample_nr):
    ground_truth_path = os.path.join(data_folder,
            "prior_samples_v2/prior_sample_{}.npy".format(sample_nr))
    ground_truth = np.load(ground_truth_path)

    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()

    irregular_array_to_point_cloud(volcano_coords.numpy(),
            ground_truth, "ground_truth_{}.vtk".format(sample_nr),
            fill_nan_val=-20000.0)

if __name__ == "__main__":
    main(4)
