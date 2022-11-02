""" This submodule collects all the static strategies (path not adapted 
as data comes in), defined as part of the Cornell-IMSV collaboration. 

First drafted: 28th October 2022.

"""
import math
import heapq
import pandas as pd
from volcapy.path_planning import sample_design_locations, solve_TSP, weight_graph_with_cost
from volcapy.set_sampling import uniform_size_uniform_sample


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

    evaluated_designs, rewards, rewards_metadata, costs, paths, accuracy_metrics = [], [], [], [], [], []
    for i, design in enumerate(candidate_designs):
        print("Evaluating design nr. {} with {} locations.".format(i, len(design)))
        path, cost = solve_TSP(accessibility_graph, design, cost_fn='weight')

        # Don't waste time computing the reward if the path exceeds 
        # the budget.
        if cost > budget:
            print("Cost exceeds budget.")
            continue
        else:
            evaluated_designs.append(design)
            result = compute_blind_reward(design, belief)
            rewards.append(result['reward'])
            rewards_metadata.append(result['reward_metadata'])
            accuracy_metrics.append(compute_accuracy_metric(design, belief))
            costs.append(cost); paths.append(path)

        # Save from time to time.
        if i % 20 == 0:
            df = pd.DataFrame.from_dict(
                    {'design': evaluated_designs,
                        'reward': rewards, 'reward_metadata': rewards_metadata, 
                        'cost': costs, 'path': paths})
            df_accuracy_metric = pd.DataFrame(accuracy_metrics)
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
                    {'design': evaluated_designs,
                        'reward': rewards, 'reward_metadata': rewards_metadata, 'cost': costs, 'path': paths})
    df_accuracy_metric = pd.DataFrame(accuracy_metrics)
    df = pd.concat([df, df_accuracy_metric], axis=1)
    df.to_pickle(output_path)
    return df


def iterative_bisection_refinement(base_node, 
        global_cost_cutoff_ratio=1.0):
    # Sampler for the base designs, sampling 2 points at random and a fixed base.
    base_designs = uniform_size_uniform_sample(
            list(accessibility_graph.node), size=n_starting_designs,
            min_size=2, max_size=2, fixed_node=base_node)

    for base_design in base_designs:
        x1, x2, x3 = base_design[0], base_design[1], base_design[2]

        path1 = nx.shortest_path(accessibility_graph, x1, x2, weight='weight')
        cost1 = nx.path_weight(accessibility_graph, path1, weight='weight')
        path2 = nx.shortest_path(accessibility_graph, x1, x2, weight='weight')
        cost2 = nx.path_weight(accessibility_graph, path1, weight='weight')
        path3 = nx.shortest_path(accessibility_graph, x1, x2, weight='weight')
        cost3 = nx.path_weight(accessibility_graph, path1, weight='weight')

        cost = cost1 + cost2 + cost3
    
        # If cost already exceeded, go to the next design.
        # In fact, since we will add waypoints, the cost will necessarily increase, 
        # so we want to be quite lower than the budget.
        if cost > global_cost_cutoff_ratio * budget:
            print("Budget exceeded.")
            continue

        # Compute the global criterion.
        global_criterion = compute_global_criterion(base_design)

        paths = [p1, p2, p3]
        for path in paths:
            x_prime = sample_refinement_point(accessibility_graph, path)
            path_leg_1 = nx.path_weight(accessibility_graph, path[0], weight='weight')
            path_leg_2 = nx.path_weight(accessibility_graph, path[-1], weight='weight')

            cost_leg1 = nx.path_weight(accessibility_graph, path_leg_1, weight='weight')
            cost_leg12= nx.path_weight(accessibility_graph, path_leg_2, weight='weight')

def sample_refinement_point(accessibility_graph, path):
    raise NotImplementedError
