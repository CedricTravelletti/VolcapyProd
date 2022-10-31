""" Utilities to load Stromboli graphs.

"""
import os
import numpy as np
import networkx as nx


def load_Stromboli_graphs(data_folder):
    """ Loads the graphs defining the data collection locations on the Stromboli. 

    This function returns the graph defining the trails (and boat connections), 
    the dense graph that meshes the whole surface of the volcano as well as 
    the merging of the two, which is the one to be used in practive for evalutating 
    strategies.

    """
    # Load graphs.
    graph_surface = nx.read_edgelist(os.path.join(data_folder, "stromboli_surface.edgelist"))
    graph_trails = nx.read_edgelist(os.path.join(data_folder, "stromboli_trails.edgelist"))
    
    # The nodes are currently labeled by strings. Set them to
    # integers, so can be used to index datapoints.
    graph_surface = nx.relabel_nodes(graph_surface, lambda x: int(x))
    graph_trails = nx.relabel_nodes(graph_trails, lambda x: int(x))
    
    # Note that the nodes in the trail graph are labelled by their position 
    # in the field campaing data 'data_coords'. 
    # To match them with the nodes in the discretisation in the surface, we have to relabel them.
    data_coords_inds_insurf = np.load(os.path.join(data_folder,"niklas_data_inds_insurf.npy"))[1:] # Remove base station.

    graph_trails_insurf = nx.relabel_nodes(graph_trails, lambda i: data_coords_inds_insurf[i])
    graph_merged = nx.compose(graph_surface, graph_trails_insurf)

    # For the sake of clarity, also add the index in the surface mesh as 
    # an attribute to each node.
    labels = {node: int(node) for node in graph_merged.nodes()}
    nx.set_node_attributes(graph_merged, labels, "node_ind_in_surface_mesh")

    return graph_trails_insurf, graph_surface, graph_merged
