import os
import pickle
import numpy as np
import pandas as pd
import torch
import networkx as nx
from volcapy.path_planning import solve_TSP
from volcapy.graph.utils import compute_path_through_nodes
from volcapy.graph.algorithms import raor_G_algorithm
from helper_functions import compute_blind_wIVR, compute_mismatch, compute_accuracy_metrics


# Load Niklas data.
# data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018/"
data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018/"
output_folder = "/storage/homefs/ct19x463/Volcano_DP_results/experiments/raor_g/"

# Prepare experiment.
from volcapy.prepare_DP_experiments import prepare_experiment_data
sample_nr = 1
(trained_gp_model, ground_truth, volcano_grid,
     data_coords, G, data_feed, niklas_data_inds_insurf,
     accessibility_graph, base_station_node,
     THRESHOLD_small, THRESHOLD_big, _) = prepare_experiment_data(sample_nr)

belief = trained_gp_model
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
