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
import networkx as nx


data_std = 0.1 

def cost_unaware_wIVR(belief, budget, accessibility_graph,
        base_station_node, waypoint_node, G, data_feed,
        lower_threshold):
    """ Run the cost unaware strategy. 

    This strategy first gathers data by travelling from the base_station_node 
    to the starting node, then starts travelling adaptively by following the wIVR 
    criterion and finally walks back to base (gathering data) when the budget is reached 
    (we take into account the fact that travelling back to base has a cost).

    """
    # First leg: Ingest data from base node to starting node.
    first_leg_nodes = nx.shortest_path(accessibility_graph, base_station_node,
            waypoint_node, weight='weight')
    cost_first_leg = nx.path_weight(accessibility_graph, first_leg_nodes,
                weight='weight')
    if cost_first_leg > budget: raise ValueError("Node not reacheable.")

    G_first_leg = G[first_leg_nodes,:]
    if len(G_first_leg.shape) <= 1: G_first_leg = G_first_leg.reshape(1, -1)
    y_first_leg = data_feed(first_leg_nodes)
    belief.update(G_first_leg, y_first_leg, data_std)

    # First go to the starting station, then start optimizing from there.
    current_node = waypoint_node
    current_path = first_leg_nodes.copy()
    remaining_budget = budget - cost_first_leg

    observed_data, data_collection_nodes = [], []

    while True:
        print("Remaining budget: {} hrs.".format(remaining_budget / 60**2))
        print("Current path: {}.".format(current_path))

        # Evaluate criterion on neigbors.
        neighbors_nodes = list(accessibility_graph.neighbors(current_node))
        neighbors_ivrs = []
        valid_neighbors = []
        for node in neighbors_nodes:
            # If we do not have budget to go there, don't even try.
            additional_cost = nx.path_weight(accessibility_graph, [current_node, node],
                    weight='weight')
            if additional_cost > remaining_budget:
                continue # Skip node.

            # Neighbor is reachable.
            valid_neighbors.append(node)

            # Observation operator for candidate location.
            # Make sure that the observation operator has 
            # at least two dimensions.
            candidate_G = G[node,:]
            if len(candidate_G.shape) <= 1: candidate_G = candidate_G.reshape(1, -1)
            current_coverage = belief.coverage(lower_threshold, None)
            ivr = belief.IVR(candidate_G, data_std, 
                                    weights=current_coverage)
            neighbors_ivrs.append(ivr)
    
        # If haven't found any reachable neighbors, stop.
        if len(valid_neighbors) == 0:
            print("Budget reached.")
            break

        # Find best candidate.
        tmp_ind = np.argmax(neighbors_ivrs)
        best_node = valid_neighbors[tmp_ind]
        print("Best node found: {}.".format(best_node))
    
        # See if we have enough budget to go there. Otherwise, search 
        # is finished.
        additional_cost = nx.path_weight(accessibility_graph, [current_node, best_node],
                weight='weight')
        print("Additional cost: {} hrs.".format(additional_cost / 60**2))
        # Always keep track of the return path.
        return_path = nx.shortest_path(accessibility_graph, best_node,
                base_station_node, weight='weight')
        return_path_cost = nx.path_weight(accessibility_graph, return_path, weight='weight')
        print("Return cost: {} hrs.".format(return_path_cost / 60**2))
    
        # If adding the new point would prevent us from returning to base, 
        # then stop and return current path.
        if additional_cost + return_path_cost > remaining_budget:
            print("Budget reached.")
            break
        else:
            # Add new node to path.
            current_path.append(best_node)
            current_node = best_node
            remaining_budget -= additional_cost

            # Go there and ingest data (this is what makes the strategy dynamic.
            # Make sure that the observation operator has 
            # at least two dimensions.
            print("Ingesting new data.")
            new_G = G[current_node,:]
            if len(new_G.shape) <= 1: new_G = new_G.reshape(1, -1)
            y = data_feed(current_node)
            observed_data.append(y)
            data_collection_nodes.append(current_node)
            belief.update(new_G, y, data_std)
            print("New data ingested.")

    # Add the return leg to the path.
    return_leg_nodes = nx.shortest_path(accessibility_graph, current_path[-1],
            base_station_node, weight='weight')
    current_path.extend(
        return_leg_nodes[1:]) # We remove the start node cause already in the list.

    # Observe the data on the return leg.
    G_return_leg = G[return_leg_nodes,:]
    if len(G_return_leg.shape) <= 1: G_return_leg = G_return_leg.reshape(1, -1)
    y_return_leg = data_feed(return_leg_nodes)
    belief.update(G_return_leg, y_return_leg, data_std)

    return current_path, observed_data, data_collection_nodes, y_first_leg, first_leg_nodes, y_return_leg, return_leg_nodes
