""" Test the Randomized Anytime Orienteering (greedy) algorithm.

Conclusion: 

"""
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from volcapy.prepare_DP_experiments import prepare_experiment_data
from volcapy.graph.algorithms import raor_algorithm, raor_G_algorithm
from volcapy.path_planning import solve_TSP

import time


sample_nr = 1
(trained_gp_model, ground_truth, volcano_grid,
    accessible_data_coords, accessible_G, accessible_data_feed, niklas_data_inds_insurf,
    accessibility_graph, base_station_node,
    THRESHOLD_small, THRESHOLD_big,
    base_results_folder) = prepare_experiment_data(sample_nr, local=True)

# Simple reward: want to maximize the number of nodes in a TSP tour below budget.
def reward_fn(node_list):
    return len(node_list)


budget = 8 * 60**2 # 8 hours.

from volcapy.set_sampling import uniform_size_uniform_sample
sampler = lambda graph: uniform_size_uniform_sample(graph, sample_size=1, max_size=10, fixed_node=base_station_node)

start = time.time()
design = raor_G_algorithm(accessibility_graph, budget, reward_fn, sampler, base_station_node,
        max_iter=100)
end = time.time()
