
""" This submodule collects all the cost-unaware dynamic strategies, 
defined as part of the Cornell-IMSV collaboration. 

Here cost-unaware means that the strategies do not take the cost into account 
during optimization, they merely run back to base once the budget is exhausted 
(taking into account the return budget).

First drafted: 28th October 2022.

"""
import os
import math
import numpy as np
import pandas as pd


# TODO: Get rid of the cost function and instead accept weighted directect 
# graph, so that the cost structure is already included.
def cost_unaware_wIVR(belief, budget, accessibility_graph,
        base_station_node, starting_node):
    # First go to the starting station, then start optimizing from there.
    current_node = starting_node
    current_path = [base_station_node, current_node]
    cost_first_leg = nx.path_weight(accessibility_graph, current_path,
                weight='weight')
    remaining_budget = budget - cost_first_leg

    observed_data = []

    while True:
        print("Remaining budget: {} hrs.".format(remaining_budget / 60**2))
        print("Current path: {}.".format(current_path))

        # Evaluate criterion on neigbors.
        neighbors_nodes = accessibility_graph.neigbors(current_node)
        for node in neighbors_nodes:
            # Observation operator for candidate location.
            # Make sure that the observation operator has 
            # at least two dimensions.
            candidate_G = self.G[ind,:]
            if len(candidate_G.shape) <= 1: candidate_G = candidate_G.reshape(1, -1)
            ivr = self.gp.IVR(candidate_G, self.data_std, 
                                    weights=self.current_coverage)
            neighbors_ivrs.append(ivr)
    
        # Find best candidate.
        tmp_ind = np.argmax(neighbors_ivrs)
        best_node = neighbors_nodes[tmp_ind]
    
        # See if we have enough budget to go there. Otherwise, search 
        # is finished.
        additional_cost = nx.path_weight(accessibility_graph, [current_node, best_node],
                weight='weight')
        # Always keep track of the return path.
        return_path = nx.shortest_path(accessibility_graph, best_node,
                base_station_node, weight='weight')
        return_path_cost = nx.path_weight(accessibility_graph, return_path, weight='weight')
    
        # If adding the new point would prevent us from returning to base, 
        # then stop and return current path.
        if additional_cost + return_path_cost > remaining_budget:
            break
        else:
            # Add new node to path.
            current_path.append(best_node)
            current_node = best_node
            remaining_budget -= additional_cost

            # Go there and ingest data (this is what makes the strategy dynamic.
            # Make sure that the observation operator has 
            # at least two dimensions.
            G = self.G[current_node,:]
            if len(G.shape) <= 1: G = G.reshape(1, -1)
            y = data_feed(current_node)
            observed_data.append(y)
            belief.update(G, y, data_std)

    return current_path, observed_data
