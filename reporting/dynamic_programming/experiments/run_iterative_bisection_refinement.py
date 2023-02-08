""" Run the iterative bisection refinement algorithm on the volcano problem.

"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import networkx as nx
from volcapy.path_planning import solve_TSP
from volcapy.graph.utils import compute_path_through_nodes
from volcapy.graph.algorithms import raor_G_algorithm
from helper_functions import prepare_helper_functions


output_folder = "/storage/homefs/ct19x463/Volcano_DP_results/experiments/iterative_bisection_refinement/"

compute_blind_wIVR, compute_mismatch, compute_accuracy_metrics = prepare_helper_functions(local=False)

# Prepare experiment.
from volcapy.prepare_DP_experiments import prepare_experiment_data
sample_nr = 1
(trained_gp_model, ground_truth, volcano_grid,
     data_coords, G, data_feed, niklas_data_inds_insurf,
     accessibility_graph, base_station_node,
     THRESHOLD_small, THRESHOLD_big, _) = prepare_experiment_data(sample_nr, local=False)

belief = trained_gp_model
budget = 60 * 60 * 8 # 8 hours.  

evaluated_designs, rewards, rewards_metadata, costs, paths, accuracy_metrics = [], [], [], [], [], []

from volcapy.strategy.static_strategies import iterative_bisection_refinement

n_starting_designs = 10
n_generations = 6
n_mutations = 3
generations_paths, generations_designs, generations_costs = iterative_bisection_refinement(accessibility_graph, base_station_node, budget,
        n_starting_designs, n_generations, n_mutations,
        max_mutations_iter=100,
        cost_growth_factor=2.0)

# Flatten the generation level.
generations_nr = [len(generations_costs[i]) * [i] for i in range(len(generations_costs))]
generations_paths = [x for x in sublist for sublist in generations_paths]
generations_designs = [x for x in sublist for sublist in generations_designs]
generations_costs = [x for x in sublist for sublist in generations_costs]

# Put everything in a dataframe.
df = pd.DataFrame.from_dict(
        {'generation': generations_nr, 'design': generations_designs, 'path': generations_paths, 'cost': generations_costs})                                                         
df.to_pickle(os.path.join(output_folder, "generations_ibr.pkl"))
