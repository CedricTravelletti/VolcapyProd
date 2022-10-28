""" Path-specific functions that help in the computation of data collection plans.

"""
import networkx as nx


def sample_design_locations(accessibility_graph, n_samples, node_weights=None, size_sampler=None):
    """ Given a graph of allowed observations locations, 
    generates random sets of design locations.

    The function allows to give more weights to certain nodes (so that they are sampled 
    with higher probability) and to choose how the sizes of the sampled set are 
    chosen. 

    By default, all nodes are sampled with equal probability, and set sizes are sampled 
    uniformly at random between 1 and the total number of nodes.

    Parameters
    ----------
    accessibility_graph
    n_samples: int
        Number of designs to sample.
    node_weights: array (n_nodes), defaults to None
        Element at index i defines the weight for graph node i.
        If not provided, then uniform weighting.
    size_sampler: function()
        Function that returns a random size at each call.

    Returns
    -------
    designs: List[design]
        List of sampled designs. Each element is a list defining the node labels
        of the design nodes (in the accessibility_graph).

    """


# TODO: Should this be part of a class defining the candidate data collection graph?
def solve_TSP(accessibility_graph, design_nodes, cost_fn):
    """ Given a set of locations, find the minimal cost path 
    connecting them by solving the Travelling Salesman Problem.

    Note: the divorced TSP problem can be solved using the procedure 
    desribed in https://cs.stackexchange.com/questions/43549/what-tsp-variant-doesnt-return-to-start-point.

    Parameters
    ----------
    accessibility_graph: networkx.Graph
    designs_nodes: List[int]
        List of node labels belonging to the design.
    cost_fn: function(edge)
        Function that computes the cost associated to a given edge.
        See volcapy.graph.cost_functions for examples.

    Returns
    -------
    path: List[int]
        List of ordered node indices (in the data space graph) defining the shortest 
        path.
    cost: float
        Cost of the shortest path.

    """
    # Generate the graph that describes the travelling salesman problem.
    TSP_graph = generate_TSP_graph(accessibility_graph, design_nodes, cost_fn)
    shortest_path = nx.algorithms.approximation.christofides(TSP_graph)
    cost = nx.path_weight(TSP_graph, shortest_path, weight='weight')
    return shortest_path, cost

def generate_TSP_graph(accessibility_graph, nodes_inds, cost_fn):
    """ Given a subset of nodes in the accessibility_graph, generate 
    graph that describes the travelling salesman problem between these nodes. 
    That is, generate the complete graph of shortes paths between the nodes.

    """
    G = nx.Graph()
    # Add nodes iteratively.
    for i, node_ind in enumerate(nodes_inds):
        # Loop over folowing nodes to connect them with shortes path.
        # Only consider following ones, so we don't double count paths.
        for end_node in nodes_inds[i+1:]:
            shortest_path_length = nx.shortest_path_length(accessibility_graph,
                    source=node_ind, target=end_node, weight=cost_fn)
            G.add_edge(node_ind, end_node, weight=shortest_path_length)
    return G
