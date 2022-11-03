""" Functions to randomly sample subsets from a given set. This is used to generate 
candidate experimental designs. 

"""
import numpy as np


def uniform_bernoulli_sample(input_set, size):
    """ Sample set by including each element of the input set 
    with probability 0.5 in the output set.

    Note that this type of sampling should be avoided in most applications, 
    since it tends to mostly samples sets of size input_set / 2, which 
    is usually too big in practice.

    Parameters
    ----------
    input_set: np.array

    Returns
    -------
    sampled_sets: List[np.array]
        List of sampled subarrays.

    """
    # Sample sizes.
    sizes = np.random.binomial(n=len(input_set), p=.5, size=size)
    sampled_sets = [np.random.choice(input_set, size=size, replace=False) for size in sizes]
    return sampled_sets

def uniform_size_uniform_sample(input_set, size, min_size=1, max_size=None, fixed_node=None):
    """ Sample set by first uniformly sampling a size in [1, input_set_size] 
    and then uniformly sampling a subset of that size.

    Parameters
    ----------
    input_set: np.array
    size
    min_size
    max_size
    fixed_node

    Returns
    -------
    sampled_sets: List[np.array]
        List of sampled subarrays.

    """
    if max_size is None:
        high_bound = len(input_set) + 1
    else: high_bound = max_size + 1

    # Sample sizes uniformly.
    sizes = np.random.randint(low=min_size, high=high_bound, size=size)
    # Sample random points and add the fixed node.
    sampled_sets = [np.random.choice(input_set, size=size, replace=False) for size in sizes]

    # If a base node has been specified, add it.
    if fixed_node is not None:
        sampled_sets = [np.concatenate([
            np.array([fixed_node]), s]) for s in sampled_sets]

    # If only one sample was requested, then do not return a list.
    if len(sampled_sets) == 1:
        sampled_sets = sampled_sets[0]
    return sampled_sets
