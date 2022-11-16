""" Submodule containing graph-related algorithms (optimization, ...).

"""
import multiprocessing
import numpy as np
import networkx as nx
from functools import partial


def recursive_greedy_SOP(graph, budget, start_node, end_node, reward_fn, max_recursion_depth):
    """ Solve the SOP problem using the recursive greedy algorithm from Chekuri and Pal (2005).

    """
    unique_weights = set([d['weight'] for *_, d in graph.edges(data=True)])
    unique_weights = np.sort(list(set(np.round(list(unique_weights)))))

    def objective_fn(path, base_set):
        return reward_fn(np.concatenate([path, base_set]))

    def _greedy_subroutine(start_node, end_node, budget, base_set, max_recursion_depth):
        print("Called, depth {}".format(max_recursion_depth))

        path = [start_node, end_node]
        if max_recursion_depth == 0: return path

        best_reward = objective_fn(path, base_set)
        print("Current best reward: {}.".format(best_reward))

        # No need to consider points that are further than what the budget allows.
        ego_start = nx.ego_graph(graph, start_node,
                radius=budget, center=False, distance='weight')
        ego_end = nx.ego_graph(graph, end_node,
                radius=budget, center=False, distance='weight')
        ego_intersection = nx.intersection(ego_start, ego_end)

        for candidate_node in list(ego_intersection.nodes)[:5]:
            allowed_lengths = np.linspace(1, budget, 4)
            for length in allowed_lengths:
                # Check if proposed budget allocation allows 
                # further calls. Otherwise skip.
                length1 = nx.shortest_path_length(graph, start_node, candidate_node, 'weight')
                length2 = nx.shortest_path_length(graph, candidate_node, end_node, 'weight')
                if (length1 > length) or (length2 > budget - length):
                    continue

                # If feasible and reached the bottom of the recursion.
                # I.e. if next call would be zero.
                if max_recursion_depth == 1:
                    path = [start_node, candidate_node, end_node]
                else:
                    path1 = _greedy_subroutine(
                            start_node, candidate_node, length, base_set, max_recursion_depth - 1)
                    augmented_set = np.concatenate([base_set, path1])
                    path2 = _greedy_subroutine(
                            candidate_node, end_node, budget - length,
                            augmented_set, max_recursion_depth - 1)

                    # Note that we have to be careful to only add the middlepoint once.
                    path = np.concatenate([path1[:-1], path2])

                reward = objective_fn(path, base_set)
                if reward > best_reward:
                    best_reward = reward
                    best_path = path
        return path

    best_path = _greedy_subroutine(start_node, end_node, budget, [], max_recursion_depth)
    return best_path

def reward_fn(path):
    return len(path)

def objective_fn(path, base_set):
    return reward_fn(np.concatenate([path, base_set]))

def recursive_greedy_SOP_parallel(
        graph, budget, start_node, end_node, max_recursion_depth, n_procs=4):
    """ Solve the SOP problem using the recursive greedy algorithm from Chekuri and Pal (2005).

    """
    # Highest level of the recutions gets parallelized.
    ego_start = nx.ego_graph(graph, start_node,
            radius=budget, center=False, distance='weight')
    ego_end = nx.ego_graph(graph, end_node,
            radius=budget, center=False, distance='weight')
    ego_intersection = nx.intersection(ego_start, ego_end)

    candidate_list = list(ego_intersection.nodes)[:5]

    pool = multiprocessing.Pool(n_procs)
    paths = pool.map(partial(
        highest_level_subroutine,
        budget=budget, graph=graph,
        start_node=start_node, end_node=end_node,
        max_recursion_depth=max_recursion_depth,
        objective_fn=objective_fn), candidate_list)

    return paths


def _greedy_subroutine(graph, start_node, end_node,
        budget, base_set, max_recursion_depth, objective_fn):
    print("Called, depth {}".format(max_recursion_depth))

    path = [start_node, end_node]
    if max_recursion_depth == 0: return path

    best_reward = objective_fn(path, base_set)
    print("Current best reward: {}.".format(best_reward))

    # No need to consider points that are further than what the budget allows.
    ego_start = nx.ego_graph(graph, start_node,
            radius=budget, center=False, distance='weight')
    ego_end = nx.ego_graph(graph, end_node,
            radius=budget, center=False, distance='weight')
    ego_intersection = nx.intersection(ego_start, ego_end)

    for candidate_node in list(ego_intersection.nodes)[:5]:
        allowed_lengths = np.linspace(1, budget, 4)
        for length in allowed_lengths:
            # Check if proposed budget allocation allows 
            # further calls. Otherwise skip.
            length1 = nx.shortest_path_length(graph, start_node, candidate_node, 'weight')
            length2 = nx.shortest_path_length(graph, candidate_node, end_node, 'weight')
            if (length1 > length) or (length2 > budget - length):
                continue

            # If feasible and reached the bottom of the recursion.
            # I.e. if next call would be zero.
            if max_recursion_depth == 1:
                path = [start_node, candidate_node, end_node]
            else:
                path1 = _greedy_subroutine(
                        graph, start_node, candidate_node,
                        length, base_set, max_recursion_depth - 1,
                        objective_fn)
                augmented_set = np.concatenate([base_set, path1])
                path2 = _greedy_subroutine(
                        graph, candidate_node, end_node,
                        budget - length,
                        augmented_set, max_recursion_depth - 1, objective_fn)

                # Note that we have to be careful to only add the middlepoint once.
                path = np.concatenate([path1[:-1], path2])

            reward = objective_fn(path, base_set)
            if reward > best_reward:
                best_reward = reward
                best_path = path
    return path

def highest_level_subroutine(
        candidate_node, budget, graph, start_node, end_node, max_recursion_depth,
        objective_fn):

    base_set = [start_node, end_node]
    best_reward = reward_fn(base_set)

    allowed_lengths = np.linspace(1, budget, 20)
    for length in allowed_lengths:
        # Check if proposed budget allocation allows 
        # further calls. Otherwise skip.
        length1 = nx.shortest_path_length(graph, start_node, candidate_node, 'weight')
        length2 = nx.shortest_path_length(graph, candidate_node, end_node, 'weight')
        if (length1 > length) or (length2 > budget - length):
            continue

        path1 = _greedy_subroutine(
                        graph, start_node, candidate_node, length, base_set, max_recursion_depth - 1,
                        objective_fn)
        augmented_set = np.concatenate([base_set, path1])
        path2 = _greedy_subroutine(
                        graph, candidate_node, end_node, budget - length,
                        augmented_set, max_recursion_depth - 1,
                        objective_fn)

        # Note that we have to be careful to only add the middlepoint once.
        path = np.concatenate([path1[:-1], path2])

        reward = objective_fn(path, base_set)
        if reward > best_reward:
            best_reward = reward
            best_path = path
    return path
