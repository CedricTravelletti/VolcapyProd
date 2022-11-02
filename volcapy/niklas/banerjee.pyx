#cython: language_level=3
# File: banerjee.py, Author: Cedric Travelletti, Date: 16.01.2019.
""" Implements the Banerjee formula for the gravitational field froduced by a
parallelepiped of uniform density.
"""
from libc.math cimport log, atan, sqrt, fabs
from cpython cimport array

def banerjee(xh, xl, yh, yl, zh, zl,
        x_data, y_data, z_data):
    return cbanerjee(xh, xl, yh, yl, zh, zl, x_data, y_data, z_data)

cdef double cbanerjee(double xh, double xl, double yh, double yl, double zh, double zl,
        double x_data, double y_data, double z_data):
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
    zh: float
        Same but along z dimension.
    zl: float
        Same but along z dimension.
    x_data: float
        x coordinate (in meters) of the location at which we measure the field.
    y_data: float
    z_data: float

    """
    cdef int i, j, l
    cdef double dx, dy, dz
    cdef int sign

    # Generate the different combinations we need.
    deltas_x = [xh - x_data, xl - x_data]
    deltas_y = [yh - y_data, yl - y_data]
    deltas_z = [zh - z_data, zl - z_data]

    cdef double B = 0
    for i, dx in enumerate(deltas_x):
        for j, dy in enumerate(deltas_y):
            for l, dz in enumerate(deltas_z):
                sign = (-1)**(i + j + l)
                B += _banerjee(dx, dy, dz, sign)
    return B

cdef double _banerjee(double x, double y, double z, int sign):
    """ Helper function for readability.
    """
    cdef double x_term
    cdef double y_term
    cdef double z_term
    cdef double TOL = 1e-6
    cdef double R = sqrt(x**2 + y**2 + z**2)

    # Check for divisions by zero and compute limits manually.
    if fabs(z - 0.0) < TOL:
        z_term = 0.0
        if fabs(x - 0.0) < TOL:
            x_term = 0.0
        else:
            x_term = sign*x * log((R + y) / (R - y))

        if fabs(y - 0.0) < TOL:
            y_term = 0.0
        else:
            y_term = sign*y * log((R + x) / (R - x))

    else:
        z_term = - 2*sign*z* atan(x * y / (R * z))
        x_term = sign*x * log((R + y) / (R - y))
        y_term = sign*y * log((R + x) / (R - x))

    return(x_term + y_term + z_term)
