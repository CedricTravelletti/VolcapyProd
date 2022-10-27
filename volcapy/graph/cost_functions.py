""" Cost functions for path on data-collection graphs.

"""
import numpy as np


def tobler_hiking_pace(slope):
    """ Returns the pace for walking on the given slope, 
    according to Tobler's model.

    Tobler, W. (1993). Three Presentations on Geographical Analysis and Modeling: Non- Isotropic Geographic Modeling; 
    Speculations on the Geometry of Geography; and Global Spatial Analysis (93-1).
    UC Santa Barbara: National Center for Geographic Information and Analysis.

    
    Parameters
    ----------
    slope: float
        vertical/horizontal distance
    
    Returns
    -------
    pace: float
        Pace in seconds per meters.
    
    """
    return 0.6 * np.exp(3.5 * np.abs(slope + 0.05))

def walking_cost_fn_edge(start_node, end_node, edge_attrs):
    """ Cost function for walking from node to node along a given edge. The cost of walking 
    a given distance is modelled using the Tobler hiking function. 
    If the path is not on a trail, a correction factor is applied.

    This function is not supposed to be used in isolation, but should 
    be provided to networkx.shortest_path and should be used with the data 
    collection graphs as specified by the volcapy package.

    One can also completely override the cost computation by providing 
    'path_attribute': 'override' and 'cost_override': cost so as to 
    manually specify the cost.
    If the 'path_attribute' is not set tor 'trail', a scaling factor 
    of 5 / 3 will be applied to the travel pace to account for off-trail 
    walking.

    Parameters
    ----------
    start_node: int
    end_node: int
    edge_attrs: dict
    ----------

    Returns
    -------
    cost: float
        Cost to travel from start_node to end_node.

    """
    hdist, vdist, path_attr = edge_attrs['hdist'], edge_attrs['vdist'], edge_attrs['path_attribute']

    if path_attr == 'override':
        return edge_attrs['cost_override']
    elif path_attr == 'trail': scale_factor = 1.0
    else: scale_factor = 5 / 3
    
    slope = vdist / hdist
    pace = tobler_hiking_pace(slope)
    dist = np.sqrt(hdist**2 + vdist**2)
    time = dist * pace
    return scale_factor * time
