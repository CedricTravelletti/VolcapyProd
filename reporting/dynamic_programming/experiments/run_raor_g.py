import os
import pickle
import numpy as np
import pandas as pd
import torch
import networkx as nx
from volcapy.grid.grid_from_dsm import Grid
from volcapy.uq.set_estimation import vorobev_expectation_inds
from volcapy.path_planning import sample_design_locations, solve_TSP, weight_graph_with_cost
from volcapy.graph.utils import compute_path_through_nodes
from volcapy.graph.algorithms import raor_G_algorithm


# Load Niklas data.
# data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018/"
data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018/"
output_folder = "/storage/homefs/ct19x463/Volcano_DP_results/experiments/raor_g/"

data_coords = np.load(os.path.join(data_folder,"niklas_data_coords_corrected_final.npy"))
data_coords_inds_insurf = np.load(os.path.join(data_folder,"niklas_data_inds_insurf.npy"))[1:] # Remove base station.
surface_data_coords = np.load(os.path.join(data_folder,"surface_data_coords.npy"))
G_full_surface = torch.from_numpy(np.load(os.path.join(data_folder,"F_full_surface.npy")))

grid = Grid.load(os.path.join(data_folder, "grid.pickle"))                                             
volcano_coords = torch.from_numpy(grid.cells).float().detach() 

from volcapy.graph.loading import load_Stromboli_graphs
graph_trails_insurf, graph_surface, graph_merged = load_Stromboli_graphs(data_folder) 

# Prepare experiment.
from volcapy.prepare_DP_experiments import prepare_experiment_data
sample_nr = 1
(trained_gp_model, ground_truth, volcano_grid,
     data_coords, G, data_feed, niklas_data_inds_insurf,
     accessibility_graph, base_station_node,
     THRESHOLD_small, THRESHOLD_big, _) = prepare_experiment_data(sample_nr)

# Define the (blind) critetion to optimize.
lower_thresold = torch.Tensor([2600.0]).to('cuda')
data_std = 0.1 

G_full_surface = torch.from_numpy(G)

def compute_blind_wIVR(design, belief):
    belief.rewind(0)
    # Get the observation operator corresponding to the design.                     
    G = G_full_surface[np.array(design), :]                                                                                                                
    current_coverage = belief.coverage(lower_thresold, None)                        
    variance_reduction, _ = belief.covariance.compute_VR(G, data_std)
    belief.rewind(0)
    wIVR = torch.sum(variance_reduction * current_coverage)
    IVR = torch.sum(variance_reduction)

    return {'reward': wIVR.item(), 'reward_metadata': {'IVR': IVR.item()}}
                                                                                  
def compute_mismatch(ground_truth, estimated_excursion_inds):  
          true_excursion_inds = (ground_truth >= lower_thresold.cpu()).nonzero()[:, 0]       
                                                                                  
          # Compute mismatch.                                                     
          mismatch = np.zeros(ground_truth.shape[0])                            
          mismatch[estimated_excursion_inds] = 1                                        
          tmp = np.zeros(ground_truth.shape[0])                                 
          tmp[true_excursion_inds] = 2                                            
          mismatch = mismatch + tmp                                               
                                                                                  
          excu_size = true_excursion_inds.shape[0] + 1                            
          p_false_pos = 100 * np.sum(mismatch == 1) / excu_size                   
          p_false_neg = 100 * np.sum(mismatch == 2) / excu_size                   
          p_correct = 100 * np.sum(mismatch == 3) / excu_size                     
                                                                                  
          return (p_false_pos, p_false_neg, p_correct)          

def compute_accuracy_metrics(ground_truth, design, belief, already_updated=False):
    if not torch.is_tensor(ground_truth): ground_truth = torch.from_numpy(ground_truth)
    if already_updated is False:
        # Get the data.
        # Get the observation operator corresponding to the design.                     
        G = G_full_surface[np.array(design), :]
        y = data_feed(np.array(design))
    
        # Add the data.
        belief.update(G, y, data_std)
    
    # Compute the new variance and coverage.
    variance_new = belief.covariance.extract_variance().cpu()
    coverage_new = belief.coverage(lower_thresold, None).cpu()
    
    # Residual weighted integrated variance.
    wIV_new = torch.sum(variance_new * coverage_new)
    IBV_final = torch.sum(coverage_new * (1 - coverage_new))
    
    # Error with plug-in estimate.
    post_mean = belief.mean_vec.cpu()
    vorobev_excursion_inds = vorobev_expectation_inds(coverage_new)
    p_false_pos, p_false_neg, p_correct = compute_mismatch(ground_truth, vorobev_excursion_inds)

    if already_updated is False:
        # Return GP to original state.
        belief.rewind(0)
    return {'wIV_final': wIV_new.item(),
            'IBV_final': IBV_final.item(), 'p_false_pos_final': p_false_pos, 'p_false_neg_final': p_false_neg, 'p_correct_final': p_correct}

belief = trained_gp_model

from volcapy.strategy.cost_unaware_dynamic_strategies import cost_unaware_wIVR

budget = 60 * 60 * 8 # 8 hours.  


evaluated_designs, rewards, rewards_metadata, costs, paths, accuracy_metrics = [], [], [], [], [], []

# Sample a base design with less than 10 nodes.
from volcapy.set_sampling import uniform_size_uniform_sample
init_design_sampler = lambda graph: uniform_size_uniform_sample(graph, sample_size=1, max_size=10, fixed_node=base_station_node)

reward_fn = lambda x: compute_blind_wIVR(x, belief)['reward']

feasible_designs = raor_G_algorithm(accessibility_graph, budget,
        reward_fn, init_design_sampler, base_station_node,
        max_iter=1e6, n_starting_designs=20)

# Compute the a-posterior metrics.
for i, design in enumerate(feasible_designs):
    print("Evaluating design nr. {} with {} locations.".format(i, len(design)))
    ordered_design, cost = solve_TSP(accessibility_graph, design, cost_fn='weight')
    print(ordered_design)
    print(cost)
    full_path, cost = compute_path_through_nodes(accessibility_graph, ordered_design, cost_fn='weight')
    print(full_path)
    print(cost)

    # Don't waste time computing the reward if the path exceeds 
    # the budget.
    if cost > budget:
        print("Cost exceeds budget.")
        continue

    # Arrange results to be saved.
    paths.append(full_path)
    evaluated_designs.append(design)
    costs.append(cost)
        
    accuracy_metric = compute_accuracy_metrics(ground_truth, design=full_path, belief=belief, already_updated=False)
    accuracy_metrics.append(accuracy_metric)
    
    # Have to reset belief to compute blind reward.
    blind_reward = compute_blind_wIVR(full_path, belief)
    
    rewards.append(blind_reward['reward'])                      
    rewards_metadata.append(blind_reward['reward_metadata'])
    
    df = pd.DataFrame.from_dict(
        {'design': evaluated_designs, 'reward': rewards, 'reward_metadata': rewards_metadata,
         'cost': costs, 'path': paths})                                                         
    df_accuracy_metric = pd.DataFrame(accuracy_metrics)                                                                           
    df = pd.concat([df, df_accuracy_metric], axis=1)
    
    df.to_pickle(os.path.join(output_folder, "results_raor_g.pkl"))
    
    # Rewind for the next.
    belief.rewind(0)
