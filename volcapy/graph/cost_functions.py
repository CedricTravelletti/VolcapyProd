""" Cost functions for path on data-collection graphs.

"""
import math
import numpy as np
from numba import float_, bool_, jit


@jit(float_(float_), nopython=True)
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
    return 0.6 * math.exp(3.5 * abs(slope + 0.05))

def _walking_cost_fn_edge(start_node, end_node, edge_attrs, symmetric=False):
    """ Helper function, see documentation of walking_cost_fn_edge.

    Parameters
    ----------
    symmetric: bool, defaults to False.
        If True, the negative slopes are treated as positive ones.

    """
    hdist, vdist, path_attr = edge_attrs['hdist'], edge_attrs['vdist'], edge_attrs['path_attribute']

    if path_attr == 'override':
        return edge_attrs['cost_override']
    elif path_attr == 'trail':
        return _walking_cost_fn_edge_jit(vdist, hdist, symmetric, scale_factor=1.0)
    else: 
        return _walking_cost_fn_edge_jit(vdist, hdist, symmetric, scale_factor=1.6666666666666667)

@jit(float_(float_, float_, bool_, float_), nopython=True)
def _walking_cost_fn_edge_jit(vdist, hdist, symmetric, scale_factor):
    slope = vdist / hdist
    if symmetric is True:
        slope = abs(slope)
    pace = tobler_hiking_pace(slope)
    dist = math.sqrt(hdist**2 + vdist**2)
    time = dist * pace
    return scale_factor * time

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
    return _walking_cost_fn_edge(start_node, end_node, edge_attrs, symmetric=False)

def symmetric_walking_cost_fn_edge(start_node, end_node, edge_attrs):
    """ Same as walking_cost_fn_edge, but ignores negative slopes (treats them 
    as positive). 

    This function is used to make the travelling salesman problem symmetric, 
    which allows to use powerful approximate algorithms (Christofides 1976).

    """
    return _walking_cost_fn_edge(start_node, end_node, edge_attrs, symmetric=True)
