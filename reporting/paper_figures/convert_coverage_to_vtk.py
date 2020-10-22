import os
import torch
import numpy as np
from volcapy.grid.grid_from_dsm import Grid
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud, _array_to_point_cloud
from volcapy.plotting.vtkutils import array_to_vector_cloud


data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"

static_results_folder = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/static/"



def main(sample_nr):
    results_folder_IVR = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/IVR_results/sample_{}/".format(sample_nr)
    results_folder_wIVR = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/wIVR_results/sample_{}/".format(sample_nr)
    
    infill_folder = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/INFILL_results/sample_{}/".format(sample_nr)
    
    ground_truth_path = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/post_samples/post_sample_{}.npy".format(sample_nr)

    reskrig_samples_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018/reskrig_samples"

    # Load
    G = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_full_surface.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()

    ground_truth = np.load(ground_truth_path)

    # Load coverage at the end of the process.
    final_coverage_wIVR = np.load(os.path.join(
            results_folder_wIVR, "coverage_90.npy"))

    # Find Vorob'ev
    # Find the Vorob'ev threshold.
    coverage = final_coverage_wIVR
    vorobev_volume = np.sum(coverage)
    def f(x):
        return np.sum(coverage > x) - vorobev_volume

    from scipy.optimize import brentq
    vorobev_threshold = brentq(f, 0.0, 1.0)

    # Plot estimated excursion set using coverage function.
    est_excursion_inds = coverage > vorobev_threshold

    # Save in array.
    vorobev_est = np.zeros(volcano_coords.shape[0])
    vorobev_est[est_excursion_inds] = 1

    irregular_array_to_point_cloud(volcano_coords.numpy(),
            final_coverage_wIVR,
            os.path.join(results_folder_wIVR,
                    "vorobev_{}.vtk".format(sample_nr)),
            fill_nan_val=-20000.0)

    irregular_array_to_point_cloud(volcano_coords.numpy(),
            final_coverage_wIVR,
            os.path.join(results_folder_wIVR,
                    "final_coverage_wIVR_{}.vtk".format(sample_nr)),
            fill_nan_val=-20000.0)

    irregular_array_to_point_cloud(volcano_coords.numpy(),
            ground_truth,
            os.path.join(results_folder_wIVR,
                    "ground_truth_{}.vtk".format(sample_nr)),
            fill_nan_val=-20000.0)

if __name__ == "__main__":
    main(8)
