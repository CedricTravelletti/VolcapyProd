import numpy as np
import networkx as nx


def compute_path_through_nodes(graph, nodes, cost_fn='weight'):
    """ Given a list of nodes (ordered) compute the details (visited nodes) 
    of the path that connects them.

    """
    path_legs = [nx.shortest_path(graph, nodes[i], nodes[i+1], weight=cost_fn) for i in range(len(nodes) - 1)]
    path_costs = [nx.shortest_path_length(graph, nodes[i], nodes[i+1], weight=cost_fn) for i in range(len(nodes) - 1)]

    full_path = np.concatenate([x[:-1] for x in path_legs])
    full_path = np.concatenate(full_path, np.array([path_legs[-1][-1]]))
    full_cost = np.sum(path_costs)

    return full_path, full_cost
