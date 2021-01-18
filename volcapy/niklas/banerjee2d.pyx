# File: banerjee2d.py, Author: Cedric Travelletti, Date: 14.01.2021.
""" Benerjee formula for a two dimensional cell. Used for 2D diagnostics.

"""
from libc.math cimport log, atan, sqrt
from cpython cimport array

def banerjee(double xh, double xl, double yh, double yl,
        double x_data, double y_data):
    """ Returns the gravity field (measured at a data point) produced by a
    parallelepiped of uniform unit density.

    Parameters
    ----------
    xh: float
        Location (in meters) of the upper (maximal x) corner of the
        parallelepiped.
    xl: float
        Same, but lower corner.
    yh: float
        Same but along y dimension.
    yl: float
        Same but along y dimension.
    x_data: float
        x coordinate (in meters) of the location at which we measure the field.
    y_data: float

    """
    cdef int i, j
    cdef double dx, dy
    cdef int sign

    # Generate the different combinations we need.
    deltas_x = [xh - x_data, xl - x_data]
    deltas_y = [yh - y_data, yl - y_data]

    cdef double B = 0
    for i, dx in enumerate(deltas_x):
        for j, dy in enumerate(deltas_y):
            sign = (-1)**i * (-1)**j
            B += _banerjee(dx, dy, sign)
    return B

cdef double _banerjee(double x, double y, int sign):
    """ Helper function for readability.
    """
    cdef double R = x**2 + y**2
    return(1/2 * (sign*y * log(R) - sign*2*y + sign*2*x * atan(y / x)))
