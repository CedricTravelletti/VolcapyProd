""" Latest (03.12.2021) functions for 3D plotting.

"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from tvtk.api import tvtk


def point_cloud_to_2D_regular_grid(point_cloud, n_grid):
    """ Interpolate a (3D) point cloud representing a surface
    to a regular 2D mesh grid to plot it using VTK.

    Parameters
    ----------
    point_cloud: (n_pts, 3) array
        Coordinates of the points.
    n_grid: int
        Number of points along one dimension of the grid.

    Returns
    -------
    (X_mesh, Y_mesh, Z_mesh): (n_grid, n_grid) array
        Points and their z-value interpolated on a 2D regular 
        meshgrid. Note that compared to the numpy convention, 
        these are already transposed, so that they can be directly 
        plotted with mayavi: mlab.surf(X_mesh, Y_mesh, Z_mesh).

    """
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    X = np.linspace(x.min(), x.max(), num=n_grid)
    Y = np.linspace(y.min(), y.max(), num=n_grid)
    X_mesh, Y_mesh = np.meshgrid(X, Y)  # 2D grid for interpolation
    interp = LinearNDInterpolator(list(zip(x, y)), z)
    Z_mesh = interp(X_mesh, Y_mesh)

    return (X_mesh.T, Y_mesh.T, Z_mesh.T)

def underground_point_cloud_to_structured_grid(
        point_cloud, point_cloud_values, surface_points,
        n_grid, fill_offset=1, fill_value=-1e6):
    """ Interpolate an underground 3D point cloud to a VTK structured grid
    so that it can be plotted with VTK. 
    Due to the interpolation happening on the convex hull of the data, one should provide 
    points at the surface so that these can be used to discard the outside of the region.

    Parameters
    ----------
    point_cloud: (n_pts, 3) array
        Coordinates of the point cloud.
    point_cloud_values: (n_pts) array
        Values at each point.
    surface_points: (n_surf, 3) array
        Coordinates of points on the surface of the region.
    fill_offset: float, default=1
        How much above the surface should we start to fill with 
        fill_value.
    fill_value: float, default=-1e6
        Value to use for points outside the region.

    Returns
    -------
    tvtk.StructuredGrid

    """
    # Create an artificial surface above the real one to make 
    # sure that values outside get interpolated to the fill value.
    artificial_surface = surface_points
    artificial_surface[:, 2] = artificial_surface[:, 2] + fill_offset
    artificial_surface_vals = np.full(artificial_surface.shape[0], fill_value)

    # Add artificial surface to the real data.
    plot_coords = np.concatenate((point_cloud, artificial_surface), axis=0)
    plot_vals = np.concatenate(
            (point_cloud_values.reshape(-1), artificial_surface_vals), axis=0)

    # Create regular mesh grid.
    x, y, z = (plot_coords[:, 0], plot_coords[:, 1],
            plot_coords[:, 2])
    X = np.linspace(x.min(), x.max(), num=n_grid)
    Y = np.linspace(y.min(), y.max(), num=n_grid)
    Z = np.linspace(z.min(), z.max(), num=n_grid)
    X_mesh, Y_mesh, Z_mesh = np.meshgrid(X, Y, Z)  # 3D grid for interpolation

    # Interpolate to the grid.
    interp = LinearNDInterpolator(list(zip(x, y, z)), plot_vals)
    vals_mesh = interp(X_mesh, Y_mesh, Z_mesh)
    vals_mesh[np.isnan(vals_mesh)] = fill_value # Set the outside to a specific value.

    # Points in the format required by VTK.
    pts = np.empty(Z_mesh.shape + (3,), dtype=float)
    pts[..., 0] = X_mesh
    pts[..., 1] = Y_mesh
    pts[..., 2] = Z_mesh
    
    # We reorder the points, scalars and vectors so this is as per VTK's
    # requirement of x first, y next and z last.
    pts = pts.transpose(2, 1, 0, 3).copy()
    pts = pts.reshape(int(pts.size / 3), 3)
    scalars = vals_mesh.transpose(2, 0, 1).copy()
    
    sg = tvtk.StructuredGrid(dimensions=X_mesh.shape, points=pts)
    sg.point_data.scalars = scalars.ravel()
    sg.point_data.scalars.name = 'values'
    return sg
