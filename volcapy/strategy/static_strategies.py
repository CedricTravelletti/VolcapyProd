""" This submodule collects all the static strategies (path not adapted 
as data comes in), defined as part of the Cornell-IMSV collaboration. 

First drafted: 28th October 2022.

"""
import math
import heapq
import pandas as pd
from volcapy.path_planning import sample_design_locations, solve_TSP, weight_graph_with_cost


# TODO: Get rid of the cost function and instead accept weighted directect 
# graph, so that the cost structure is already included.
def static_blind_path_selection(
        compute_blind_reward, compute_accuracy_metric,
        belief, budget, N_refine, 
        accessibility_graph, cost_fn, design_sampler, 
        n_starting_designs,
        output_path):
    """ Compute optimal designs using static blind path selection 
    (see companion paper). Here blind means that the optimization does not look 
    at the (potentially) observed data and static means that there is no adaptation 
    as the data collection processs unfolds.

    Parameters
    ----------
    compute_blind_reward: function(design, belief)
        Function that computes the (blind) reward for a given design, 
        give the current belief. 
    belief: volcapy.update.UpdatableGP
        The prior GP belief.
    budget: float
        Maximal allowed cost.
    N_refine: int
        TODO
    accessibility_graph: nx.Graph
        A labelled graph defining the data collection landscape.
    cost_fn: function(star_node, end_node, edge_attrs)
        Function defining the cost of a given edge in the accessibility_graph.
    cost_fn: function(star_node, end_node, edge_attrs)
        Function defining the cost of a given edge in the accessibility_graph.
    design_sampler: function(n_samples)
        Function that randomly samples designs (see volcapy.set_sampling).
    n_starting_designs: int
        Number of designs to sample at beginning of the optimization process.

    Returns
    -------
    designs

    """
    # Precompute the cost of each edges to accelerate the next computations.
    accessibility_graph = weight_graph_with_cost(accessibility_graph, cost_fn)

    # Sample candidate designs.
    candidate_designs = design_sampler(list(accessibility_graph.nodes), n_starting_designs)

    rewards, costs, paths, accuracy_metrics = [], [], [], []
    for i, design in enumerate(candidate_designs):
        print("Evaluating design nr. {} with {} locations.".format(i, len(design)))
        path, cost = solve_TSP(accessibility_graph, design, cost_fn='weight')

        # Don't waste time computing the reward if the path exceeds 
        # the budget.
        if cost > budget:
            print("Cost exceeds budget.")
            continue
        else:
            rewards.append(compute_blind_reward(design, belief))
            accuracy_metrics.append(compute_accuracy_metric(design, belief))
            costs.append(cost); paths.append(path)

        # Save from time to time.
        if i % 20 == 0:
            df = pd.DataFrame.from_dict(
                    {'design': candidate_designs[:i+1],
                    'reward': rewards, 'cost': costs, 'path': paths})
            df_accuracy_metric = pd.DataFrame(accuracy_metrics).to_dict(orient="list") 
            df = pd.concat([df, df_accuracy_metric], axis=1)
            df.to_pickle(output_path)

    """
    # Get N_refine largest.
    best_designs_inds = heapq.nlargest(N_refine, range(len(rewards)), rewards.__getitem__)
    best_designs = [candidate_designs[i] for i in best_designs_inds]
    best_rewards = [rewards[i] for i in best_designs_inds]
    best_costs = [costs[i] for i in best_designs_inds]
    best_paths = [paths[i] for i in best_designs_inds]
    """
    # Save at the end
    df = pd.DataFrame.from_dict(
                    {'design': candidate_designs[:i+1],
                    'reward': rewards, 'cost': costs, 'path': paths})
    df_accuracy_metric = pd.DataFrame(accuracy_metrics).to_dict(orient="list") 
    df = pd.concat([df, df_accuracy_metric], axis=1)
    df.to_pickle(output_path)
    return df
