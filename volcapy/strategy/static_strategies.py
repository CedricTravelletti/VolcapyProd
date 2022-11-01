""" This submodule collects all the static strategies (path not adapted 
as data comes in), defined as part of the Cornell-IMSV collaboration. 

First drafted: 28th October 2022.

"""
import math
import heapq
from volcapy.path_planning import sample_design_locations, solve_TSP


# TODO: Get rid of the cost function and instead accept weighted directect 
# graph, so that the cost structure is already included.
def static_blind_path_selection(
        compute_blind_reward,
        belief, budget, N_refine, 
        accessibility_graph, cost_fn, design_sampler):
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
    candidate_designs = design_sampler(n_starting_designs)

    rewards, costs, paths = [], [], []
    for design in candidate_designs:
        path, cost = solve_TSP(accessibility_graph, design, cost_fn='weight')

        # Don't waste time computing the reward if the path exceeds 
        # the budget.
        if cost > budget:
            reward = -maht.inf
        else: reward = compute_blind_reward(design, belief)

        # Store results.
        rewards.append(reward)
        costs.append(cost)
        paths.append(cost)

    # Get N_refine largest.
    best_designs_inds = heapq.nlargest(N_refine, range(len(rewards)), rewards.__getitem__)
    best_designs = [candidate_designs[i] for i in best_designs_inds]
    best_rewards = [rewards[i] for i in best_designs_inds]
    best_costs = [costs[i] for i in best_designs_inds]
    best_paths = [paths[i] for i in best_designs_inds]

    return (best_designs, best_rewards, best_costs, best_paths)


def iterative_bisection_refinement(base_node):
    # Sampler for the base designs, sampling 2 points at random and a fixed base.
    base_sampler = 
    for i in range(n_starting_designs):
        x1, x2, x3 = base_sampler()
        path1 = nx.shortest_path(accessibility_graph, x1, x2, weight='weight')
        cost1 = nx.path_weight(accessibility_graph, path1, weight='weight')
        path2 = nx.shortest_path(accessibility_graph, x1, x2, weight='weight')
        cost2 = nx.path_weight(accessibility_graph, path1, weight='weight')
        path3 = nx.shortest_path(accessibility_graph, x1, x2, weight='weight')
        cost3 = nx.path_weight(accessibility_graph, path1, weight='weight')
    
        # If cost already exceeded, go to the next design.
        if cost1 + cost2 + cost3 > budget:
            print("Budget exceeded.")
            continue
        paths = [p1, p2, p3]
        for path in paths:
            x_prime = add_refinement_point(accessibility_graph, path)
            path_leg_1 = nx.path_weight(accessibility_graph, path[0], weight='weight')
            path_leg_2 = nx.path_weight(accessibility_graph, path[-1], weight='weight')

            cost_leg1 = nx.path_weight(accessibility_graph, path_leg_1, weight='weight')
            cost_leg12= nx.path_weight(accessibility_graph, path_leg_2, weight='weight')

