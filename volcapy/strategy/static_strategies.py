""" This submodule collects all the static strategies (path not adapted 
as data comes in), defined as part of the Cornell-IMSV collaboration. 

First drafted: 28th October 2022.

"""
import math
import heapq
import numpy as np
import pandas as pd
import networkx as nx
from volcapy.path_planning import sample_design_locations, solve_TSP, weight_graph_with_cost
from volcapy.set_sampling import uniform_size_uniform_sample


# TODO: Get rid of the cost function and instead accept weighted directect 
# graph, so that the cost structure is already included.
def static_blind_path_selection(
        compute_blind_reward, compute_accuracy_metric,
        belief, budget, N_refine, 
        accessibility_graph, design_sampler, 
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
        A weighted graph defining the data collection landscape.
    design_sampler: function(n_samples)
        Function that randomly samples designs (see volcapy.set_sampling).
    n_starting_designs: int
        Number of designs to sample at beginning of the optimization process.

    Returns
    -------
    designs

    """
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


def iterative_bisection_refinement(accessibility_graph, base_node, 
        budget,
        n_starting_designs,
        n_generations,
        n_mutations,
        compute_global_criterion,
        sample_refinement_point,
        global_cost_cutoff_ratio=1.0):
    # Sampler for the base designs, sampling 2 points at random and a fixed base.
    base_designs = uniform_size_uniform_sample(
            list(accessibility_graph.nodes), sample_size=n_starting_designs,
            min_size=2, max_size=2, fixed_node=base_node)

    # Case where we have only one design.
    if n_starting_designs == 1:
        base_designs = [base_designs]

    # Base generation contains a bunch of designs, made of three paths.
    base_generation = [
        [
            nx.shortest_path(accessibility_graph, design[0], design[1], weight='weight'),
            nx.shortest_path(accessibility_graph, design[1], design[2], weight='weight'),
            nx.shortest_path(accessibility_graph, design[2], design[0], weight='weight')
        ]
        for design in base_designs]
    generations = [base_generation]
    for k in range(n_generations):
        new_generation = []
        for design in generations[k]:
            for l in range(n_mutations):
                new_design = []
                for segment in design:
                    # Endpoints.
                    s1, s2 = segment[0], segment[-1]
                    # Sample refinement point.
                    s_prime = sample_refinement_point(accessibility_graph, segment)

                    # Replace segment by mutated segment.
                    path_leg1 = nx.shortest_path(accessibility_graph,
                            s1, s_prime, weight='weight')
                    path_leg2 = nx.shortest_path(accessibility_graph,
                            s_prime, s2, weight='weight')
                    new_design.append(path_leg1)
                    new_design.append(path_leg2)
                # Check if new design is within cost.
                cost_new_design = np.sum([
                    nx.path_weight(accessibility_graph, path_leg, weight='weight')
                    for path_leg in new_design])
                if cost_new_design < budget:
                    new_generation.append(new_design)
        # If not mutations have been generated, then stop.
        if len(new_generation) == 0:
            return generations
        else:
            generations.append(new_generation)
    return generations

def sample_refinement_point(accessibility_graph, path):
    cost_growth_factor = 1.0
    path_cost = nx.path_weight(accessibility_graph, path, weight='weight')

    start, end = path[0], path[-1]
    ## Compute graphs of nodes within a given distance.
    ego_start = nx.ego_graph(accessibility_graph, start, radius=cost_growth_factor*path_cost, center=False, distance='weight')
    ego_end = nx.ego_graph(accessibility_graph, end, radius=path_cost, center=False, distance='weight')

    # Intersect to get graph that are within given distance for both nodes.
    ego_intersection = nx.intersection(ego_start, ego_end)

    # Randomly choose one node in there as a refinement.
    refinement_node = np.random.choice(list(ego_intersection.nodes()), size=1, replace=False)[0]
    return refinement_node
