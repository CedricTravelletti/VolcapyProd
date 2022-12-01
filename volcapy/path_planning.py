""" Path-specific functions that help in the computation of data collection plans.

"""
import networkx as nx
from functools import partial
import multiprocessing


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

    Parameters
    ----------
    cost_fn: callable or string
        Either a cost function defining the cost of a given edge, or a string 
        defining which edge attribute should be used as weight/distance for that edge.

    """
    # Special case.
    if len(nodes_inds) == 1:
        G = nx.Graph()
        G.add_node(nodes_inds[0])
        return G

    # Add nodes iteratively.
    pool = multiprocessing.Pool(4)
    partial_graphs = pool.map(partial(
        _gen_TSP_graph_worker_fn,
        nodes_inds=nodes_inds, accessibility_graph=accessibility_graph,
        cost_fn=cost_fn), range(len(nodes_inds)))
    """
    for i, node_ind in enumerate(nodes_inds):
        # Loop over folowing nodes to connect them with shortest path.
        # Only consider following ones, so we don't double count paths.
        for end_node in nodes_inds[i+1:]:
            shortest_path_length = nx.shortest_path_length(accessibility_graph,
                    source=node_ind, target=end_node, weight=cost_fn)
            G.add_edge(node_ind, end_node, weight=shortest_path_length)
    """
    # Merge the resulting graphs.
    G = nx.compose_all(partial_graphs)
    return G

def _gen_TSP_graph_worker_fn(i, nodes_inds, accessibility_graph, cost_fn):
    """ Worker function for parallization of generate_TSP_graph.

    """
    G = nx.Graph()

    for end_node in nodes_inds[i+1:]:
        node_ind = nodes_inds[i]
        shortest_path_length = nx.shortest_path_length(accessibility_graph,
                    source=node_ind, target=end_node, weight=cost_fn)
        G.add_edge(node_ind, end_node, weight=shortest_path_length)

    return G

def add_node_to_TSP_graph(new_node, TSP_graph, base_graph, cost_fn):
    # Connect the new node to all the nodes belongina to the TSP graph.
    for end_node in list(TSP_graph.nodes):
        shortest_path_length = nx.shortest_path_length(base_graph,
                    source=new_node, target=end_node, weight=cost_fn)
        TSP_graph.add_edge(new_node, end_node, weight=shortest_path_length)

    return TSP_graph

def del_node_from_TSP_graph(node_to_del, TSP_graph):
    TSP_graph.remove_node(node_to_del)
    return TSP_graph

def weight_graph_with_cost(graph, cost_fn):
    """ Given a cost function and a graph, weight each edge with the corrsponding cost.
    This is useful for pre-computing costs before solving travelling salesman problems. 

    """
    weights = []
    for e in graph.edges(data=True):
        nx.set_edge_attributes(graph, values={(e[0], e[1]): cost_fn(*e)}, name = 'weight')
    return graph
