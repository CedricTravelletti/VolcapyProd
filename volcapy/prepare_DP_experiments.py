""" Prepare the data and set up the environment to run the Dynamic Programming 
experiments. 

We assume throughout that the cost function used is the symmetric versions 
of the Tobler hiking function.

"""
import os
import numpy as np
import torch
import volcapy.covariance.matern32 as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance import UpdatableGP


def prepare_experiment_data(sample_nr, local=False):
    data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
    ground_truth_folder = "/storage/homefs/ct19x463/AISTATS_results/final_samples_matern32/"
    base_results_folder = "/storage/homefs/ct19x463/Volcano_DP_results/"

    if local is True:
        data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018/"
        ground_truth_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/AISTATS_results_v2/FINAL/final_samples_matern32/"

    # Load static data.
    G = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_full_surface.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()
    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"surface_data_coords.npy"))).float()

    # Load generated data.
    ground_truth_path = os.path.join(ground_truth_folder,
            "prior_sample_{}.npy".format(sample_nr))
    ground_truth = torch.from_numpy(
            np.load(ground_truth_path))

    # --------------------------------
    # DEFINITION OF THE EXCURSION SET.
    # --------------------------------
    THRESHOLD_small = 2600.0
    THRESHOLD_big = 2500.0

    # Start at base station.
    start_ind = 4478

    # Prepare data.
    data_values = G @ ground_truth
    data_feed = lambda x: data_values[x]

    # -------------------------------------
    # GP model trained on field data.
    # -------------------------------------
    data_std = 0.1
    sigma0_matern32 = 284.66
    m0_matern32 = 2139.1
    lambda0_matern32 = 651.58
    trainded_gp_model = UpdatableGP(cl, lambda0_matern32, sigma0_matern32, m0_matern32,
            volcano_coords, n_chunks=200)

    # Load the data collection graphs.
    from volcapy.graph.loading import load_Stromboli_graphs
    graph_trails_insurf, graph_surface, accessibility_graph = load_Stromboli_graphs(data_folder)

    # The starting point for all data collection campaigns.
    base_station_node = list(graph_trails_insurf.nodes)[0]

    # Weight the edges with the walking cost, so its pre-computed 
    # once and for all.
    from volcapy.graph.cost_functions import symmetric_walking_cost_fn_edge
    from volcapy.path_planning import weight_graph_with_cost
    accessibility_graph = weight_graph_with_cost(accessibility_graph, symmetric_walking_cost_fn_edge)

    
    return (trained_gp_model, ground_truth,
            accessible_data_coords, accessible_G, accessible_data_feed,
            accessibility_graph, base_station_node,
            THRESHOLD_small, THRESHOLD_big,
            base_results_folder)