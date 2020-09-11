""" Various utility functions.

"""


def _make_column_vector(y):
    """ Make sure the data is a column vector.

    """
    if ((len(y.shape) >= 2) and (y.shape[1] == 1)):
        return y
    elif len(y.shape) == 1:
        return y.reshape(-1, 1)
    else:
        raise ValueError(
        "Shape of data vector {} is not valid. Please provide a column vector.".format(y.shape))
