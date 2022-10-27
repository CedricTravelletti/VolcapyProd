import volcapy.covariance.matern32 as cl
from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance import UpdatableGP


def prepare_experiment_data(sample_nr):
    data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
    ground_truth_folder = "/storage/homefs/ct19x463/AISTATS_results/final_samples_matern32/"
    base_results_folder = "/storage/homefs/ct19x463/Volcano_DP_results/"

    # Create output directory.
    output_folder = os.path.join(base_results_folder,
            "wIVR_no_cost_small/sample_{}".format(sample_nr))
    os.makedirs(output_folder, exist_ok=True)

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
    THRESHOLD_low = 2600.0

    # Start at base station.
    start_ind = 4478
    
    # -------------------------------------
    # Define GP model (trained on field data).
    # -------------------------------------
    data_std = 0.1
    sigma0_matern32 = 284.66
    m0_matern32 = 2139.1
    lambda0_matern32 = 651.58

    # Prepare data.
    data_values = G @ ground_truth
    data_feed = lambda x: data_values[x]

    gp_model = UpdatableGP(cl, lambda0_matern32, sigma0_matern32, m0_matern32,
            volcano_coords, n_chunks=200)

    # Load the data collection graphs.
    from volcapy.graph.loading import load_Stromboli_graphs
    graph_trails_insurf, graph_surface, graph_merged = load_Stromboli_graphs(data_folder)
    
    return (gp_model, data_coords, G, data_feed,
            graph_merged, THRESHOLD_low, start_ind, output_folder)
