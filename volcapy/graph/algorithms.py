""" Submodule containing graph-related algorithms (optimization, ...).

"""
import multiprocessing
import numpy as np
import networkx as nx
from functools import partial
from volcapy.path_planning import generate_TSP_graph, add_node_to_TSP_graph, del_node_from_TSP_graph, solve_TSP
from volcapy.set_sampling import uniform_size_uniform_sample

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

def gcb_algorithm(graph, budget, reward_fn, cost_fn):
    """ Runs the generalized cost-benefit algorithm from Zhang and Vorobeychik (2016) 
    "Submodular Optimization with Routing Constraints".

    """
    current_design = []
    feasible_nodes = list(graph.nodes).copy()
    count = 0
    while len(feasible_nodes) > 0:
        base_reward = reward_fn(current_design)
        base_cost = cost_fn(current_design)

        costs, rewards = [], []
        for node in feasible_nodes:
            cost = cost_fn(current_design + [node])
            reward = reward_fn(current_design + [node])
            costs.append(cost)
            rewards.append(reward)

        # TODO: we could implement this if we had the real cost function. 
        # problem here is that we are only working with a 3/2 approximation (Christofides).
        # We know the cost can only increase as we add more points, 
        # so already remove the ones we don't need.

        # Find best.
        costs, rewards = np.array(costs), np.array(rewards)
        deltas = (rewards - base_reward) / (costs - base_cost)
        best_node = feasible_nodes[np.argmax(deltas)]

        new_cost = cost_fn(current_design + [best_node])
        if new_cost < budget:
            current_design.append(best_node)

        # No need to consider that node anymore.
        feasible_nodes.remove(best_node)
        count += 1
        print(current_design)
        print(new_cost / 60)
    return current_design

def raor_algorithm(graph, budget, reward_fn, sampler, base_station_node):
    """ Runs the randomized anytime orienteering algorithm from Arora 
    and Scherer (2017).

    """
    nodes = np.array(list(graph.nodes))
    init_design = sampler(nodes)

    # Generate the base TSP graph.
    TSP_graph = generate_TSP_graph(graph, init_design, cost_fn='weight')

    # Compute the base cost.
    init_path = nx.algorithms.approximation.christofides(TSP_graph)
    init_cost = nx.path_weight(TSP_graph, init_path, weight='weight')

    init_reward = reward_fn(init_design)
    current_design = init_design

    # Check if satisfies budget.
    if init_cost <= budget:
        best_reward = init_reward
    else: best_reward = 0

    for i in range(3 * len(nodes)):
        print(i)
        print("Current best reward: {}.".format(best_reward))
        # Randomly sample a node.
        new_node = np.random.choice(nodes)
        
        # If already in design, then delete, else add to design.
        if new_node in current_design:
            current_design = np.delete(current_design, np.where(current_design == new_node))
            TSP_graph = del_node_from_TSP_graph(new_node, TSP_graph)
        else:
            current_design = np.append(current_design, [new_node])
            TSP_graph = add_node_to_TSP_graph(new_node, TSP_graph, graph,
                    cost_fn='weight')

        new_reward = reward_fn(current_design)

        # Solve TSP.
        new_path = nx.algorithms.approximation.christofides(TSP_graph)
        new_cost = nx.path_weight(TSP_graph, new_path, weight='weight')
        print("New cost: {} hrs.".format(new_cost / (60**2)))

        if ((new_cost <= budget) and (new_reward > best_reward)):
            best_reward = new_reward
            best_designs = current_design

    return best_design

def raor_G_algorithm(graph, budget, reward_fn, sampler, base_station_node,
        max_iter=1e6, n_starting_designs=10):
    """ Runs the randomized anytime orienteering algorithm from Arora 
    and Scherer (2017) in its greedy variant.

    Parameters
    ----------
    n_starting_designs: int, defaults to 20.

    """
    nodes = np.array(list(graph.nodes))

    # Keep a list of all the feasible designs found.
    # TODO: This time do not include the base station node alone, 
    # since will create problem (one node graph) when solving TSP.
    """
    feasible_designs = [np.array([base_station_node])] # Start with trivial design.
    feasible_designs_rewards = [reward_fn(np.array([base_station_node]))]
    """
    feasible_designs, feasible_designs_rewards = [], []

    # Before starting, seed the set of feasible designs with a bunch of two points designs.
    # Only select point within the budget.
    nodes_within_budget = np.array(list(
        nx.ego_graph(graph, base_station_node,
                radius=budget/2, center=False, distance='weight').nodes))

    for i in range(n_starting_designs):
        new_node = np.random.choice(nodes_within_budget)
        if nx.shortest_path_length(graph, base_station_node, new_node, weight='weight') <= budget:
            feasible_designs.append(np.array([base_station_node, new_node]))
            feasible_designs_rewards.append(reward_fn(np.array([base_station_node, new_node])))

    # Find a starting set that satisfies the constraints.
    init_design = sampler(nodes)

    # Generate the base TSP graph.
    TSP_graph = generate_TSP_graph(graph, init_design, cost_fn='weight')

    # Compute the base cost.
    init_path = nx.algorithms.approximation.christofides(TSP_graph)
    init_cost = nx.path_weight(TSP_graph, init_path, weight='weight')

    if init_cost <= budget:
        current_design = init_design
        best_reward = reward_fn(init_design)
        feasible_designs.append(current_design)
        feasible_designs_rewards.append(best_reward)
    # Otherwise seed with just the base station.
    else: 
        current_design = np.array([base_station_node])
        best_reward = reward_fn(current_design)

    # Generate the TSP graphs for the feasible designs.
    feasible_designs_TSP_graphs = [
            generate_TSP_graph(graph, design, cost_fn='weight')
            for design in feasible_designs]

    for i in range(3 * len(nodes)):
        if i >= max_iter: return feasible_designs
        print(i)
        print("Current best reward: {}.".format(best_reward))
        # Randomly sample a node.
        new_node = np.random.choice(nodes)
        
        # If already in design, then delete, else add to design.
        if new_node in current_design:
            current_design = np.delete(current_design, np.where(current_design == new_node))
            TSP_graph = del_node_from_TSP_graph(new_node, TSP_graph)
        else:
            current_design = np.append(current_design, [new_node])
            TSP_graph = add_node_to_TSP_graph(new_node, TSP_graph, graph,
                    cost_fn='weight')

        new_reward = reward_fn(current_design)

        # Solve TSP.
        new_path = nx.algorithms.approximation.christofides(TSP_graph)
        new_cost = nx.path_weight(TSP_graph, new_path, weight='weight')
        print("New cost: {} hrs.".format(new_cost / (60**2)))

        if new_cost <= budget:
            # Found new feasible design.
            feasible_designs.append(current_design)
            feasible_designs_rewards.append(new_reward)
            feasible_designs_TSP_graphs.append(TSP_graph)

        feasible_designs, feasible_designs_rewards, feasible_designs_TSP_graphs = refine_feasible_designs(
                new_node, graph,
                feasible_designs, feasible_designs_rewards, feasible_designs_TSP_graphs,
                budget, reward_fn)

    return feasible_designs

def refine_feasible_designs(new_node, graph,
        feasible_designs, feasible_designs_rewards, feasible_designs_TSP_graphs,
        budget, reward_fn):

    # Compute the new costs.
    feasible_designs_TSP_graphs = [
            add_node_to_TSP_graph(new_node, TSP_graph, graph,
                    cost_fn='weight')
            for TSP_graph in feasible_designs_TSP_graphs]
    # Solve TSP.
    new_paths = [nx.algorithms.approximation.christofides(TSP_graph) 
            for TSP_graph in feasible_designs_TSP_graphs]
    new_costs = np.array([nx.path_weight(TSP_graph, new_path, weight='weight')
            for new_path, TSP_graph in zip(new_paths, feasible_designs_TSP_graphs)])

    # Now return TSP graphs to their original state.
    feasible_designs_TSP_graphs = [
            del_node_from_TSP_graph(new_node, TSP_graph)
            for TSP_graph in feasible_designs_TSP_graphs]

    # Solve old TSP.
    old_paths = [nx.algorithms.approximation.christofides(TSP_graph) 
            for TSP_graph in feasible_designs_TSP_graphs]
    old_costs = np.array([nx.path_weight(TSP_graph, old_path, weight='weight')
            for old_path, TSP_graph in zip(old_paths, feasible_designs_TSP_graphs)])

    print((1 / 60**2) * old_costs)
    print((1 / 60**2) * new_costs)

    # If all exceed budget, then nothing left to do.
    if np.all(new_costs > budget):
        print("No design can be refined.")
        return feasible_designs, feasible_designs_rewards, feasible_designs_TSP_graphs

    # Otherwise, add to the best candidate.
    else:
        new_rewards = np.zeros(new_costs.shape)
        # Only compute rewards where we havent exceeded budget.
        new_rewards[new_costs <= budget] = np.array(
                [reward_fn(np.append(feasible_designs[i], [new_node]))
                    for i, x in enumerate(new_costs <= budget) if x])
    
        # Keep the one that produces the best marginal increment.
        best_design_ind = np.argmax(
                (new_rewards - feasible_designs_rewards) / new_costs)
    
        # Replace the chosen design by its modified version.
        modified_design = np.append(feasible_designs[best_design_ind], [new_node])
        feasible_designs[best_design_ind] = modified_design
        feasible_designs_rewards[best_design_ind] = new_rewards[best_design_ind]

        # Update its TSP graph.
        feasible_designs_TSP_graphs[best_design_ind] = add_node_to_TSP_graph(
                new_node, feasible_designs_TSP_graphs[best_design_ind], graph, cost_fn='weight')
    
        return feasible_designs, feasible_designs_rewards, feasible_designs_TSP_graphs
