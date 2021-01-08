""" Generate conditional realizations from the distribution at the end of the
run of a strategy.

"""
import os
import torch
import numpy as np
import volcapy.covariance.exponential as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance import UpdatableGP, UpdatableRealization
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud, _array_to_point_cloud
from volcapy.plotting.vtkutils import array_to_vector_cloud


from timeit import default_timer as timer


sample_nr = 8

data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"
results_folder_IVR = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/IVR_results/sample_{}/".format(sample_nr)
results_folder_wIVR = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/wIVR_results/sample_{}/".format(sample_nr)

infill_folder = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/INFILL_results/sample_{}/".format(sample_nr)
results_folder_INFILL = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/INFILL_results/sample_{}/".format(sample_nr)

static_results_folder = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/static/"

ground_truth_path = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/post_samples/post_sample_{}.npy".format(sample_nr)

reskrig_samples_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018/reskrig_samples"


def main():
    # Load
    G = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_full_surface.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()

    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"surface_data_coords.npy"))).float()

    ground_truth = torch.from_numpy(np.load(ground_truth_path))

    # Dictionary between the original Niklas data and our discretization.
    niklas_data_inds = torch.from_numpy(
            np.load(os.path.join(data_folder, "niklas_data_inds_insurf.npy"))).long()
    niklas_coords = data_coords[niklas_data_inds].numpy()


    # --------------------------------
    # DEFINITION OF THE EXCURSION SET.
    # --------------------------------
    THRESHOLD_low = 700.0
    excursion_inds = (ground_truth >= THRESHOLD_low).nonzero()[:, 0]

    # Load results.
    visited_inds_IVR = np.load(os.path.join(
            results_folder_IVR, "visited_inds.npy"))
    visited_inds_wIVR = np.load(os.path.join(
            results_folder_wIVR, "visited_inds.npy"))

    # Observation operators.
    # G_stacked_IVR = G[visited_inds_IVR, :]
    G_stacked_wIVR = G[visited_inds_wIVR, :]

    # Reload the GPs.
    # gpIVR = UpdatableGP.load(os.path.join(results_folder_IVR, "gp_state.pkl"))
    # gpwIVR = UpdatableGP.load(os.path.join(results_folder_wIVR, "gp_state.pkl"))
    gpINFILL = UpdatableGP.load(os.path.join(results_folder_INFILL, "gp_state.pkl"))

    # Produce posterior realization.
    for reskrig_sample_nr in range(300, 400):
        prior_realization = torch.from_numpy(np.load(
                os.path.join(reskrig_samples_folder,
                        "prior_sample_{}.npy".format(reskrig_sample_nr))))
        myReal = UpdatableRealization.bootstrap(prior_realization, G_stacked_wIVR,
                data_std=0.1, gp_module=gpINFILL)
        np.save(
                os.path.join(results_folder_wIVR,
                    "Cond_Reals_Infill/conditional_real_{}.npy".format(reskrig_sample_nr)),
                myReal._realization.detach().cpu().numpy())


if __name__ == "__main__":
    main()
