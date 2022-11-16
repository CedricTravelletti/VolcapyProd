import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from volcapy.prepare_DP_experiments import prepare_experiment_data

import time


sample_nr = 1
(trained_gp_model, ground_truth, volcano_grid,
    accessible_data_coords, accessible_G, accessible_data_feed, niklas_data_inds_insurf,
    accessibility_graph, base_station_node,
    THRESHOLD_small, THRESHOLD_big,
    base_results_folder) = prepare_experiment_data(sample_nr, local=False)

def reward_fn(node_list):
    return len(node_list)

from volcapy.graph.algorithms import recursive_greedy_SOP, recursive_greedy_SOP_parallel

start_node, end_node = 0, 0
budget = 6 * 60**2

start = time.time()
path = recursive_greedy_SOP_parallel(accessibility_graph, budget, start_node, end_node, max_recursion_depth=3)
end = time.time()

print(path)

print((end - start) / 60)
