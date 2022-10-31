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

def uniform_size_uniform_sample(input_set, size, min_size=1, max_size=None):
    """ Sample set by first uniformly sampling a size in [1, input_set_size] 
    and then uniformly sampling a subset of that size.

    Parameters
    ----------
    input_set: np.array

    Returns
    -------
    sampled_sets: List[np.array]
        List of sampled subarrays.

    """
    if max_size is None:
        max_size = len(input_set) + 1

    # Sample sizes uniformly.
    sizes = np.random.randint(low=min_size, high=max_size, size=size)
    sampled_sets = [np.random.choice(input_set, size=size, replace=False) for size in sizes]
    return sampled_sets
