""" Latest (03.12.2021) functions for 3D plotting.

"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from tvtk.api import tvtk
import plotly.graph_objects as go


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
        n_grid_x=100, n_grid_y=100, n_grid_z=35,
        fill_offset=1, fill_value=-1e6, to_vtk=True):
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
    n_grid_x: int
        Number of grid points along the x-axis.
    n_grid_y: int
        Number of grid points along the y-axis.
    n_grid_z: int
        Number of grid points along the z-axis.
    fill_offset: float, default=1
        How much above the surface should we start to fill with 
        fill_value.
    fill_value: float, default=-1e6
        Value to use for points outside the region.
    to_vtk: bool, default to True
        If True, output a VTK StructuredGrid, otherwise 
        output the meshes and meshed values in numpy format.

    Returns
    -------
    tvtk.StructuredGrid
    or
    x_mesh, y_mesh, z_mesh, vals_mesh: array

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
    X = np.linspace(x.min(), x.max(), num=n_grid_x)
    Y = np.linspace(y.min(), y.max(), num=n_grid_y)
    Z = np.linspace(z.min(), z.max(), num=n_grid_z)
    X_mesh, Y_mesh, Z_mesh = np.meshgrid(X, Y, Z)  # 3D grid for interpolation

    # Interpolate to the grid.
    interp = LinearNDInterpolator(list(zip(x, y, z)), plot_vals)
    vals_mesh = interp(X_mesh, Y_mesh, Z_mesh)
    vals_mesh[np.isnan(vals_mesh)] = fill_value # Set the outside to a specific value.

    # If only numpy output.
    if to_vtk is False:
        return X_mesh, Y_mesh, Z_mesh, vals_mesh

    # Else return a VTK structured grid.
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

# DEPRECATED: replaced by volcapy.grid.mesh_values()
def interpolate_to_plane_slice(point_cloud, point_cloud_values, altitude, n_grid):
    # Create a meshgrid on the xy axis.
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    X = np.linspace(x.min(), x.max(), num=n_grid)
    Y = np.linspace(y.min(), y.max(), num=n_grid)
    X_mesh, Y_mesh = np.meshgrid(X, Y)  # 2D grid for interpolation

    Z_mesh = altitude * np.ones(X_mesh.shape)

    # Define interpolator for the data.
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    interp = LinearNDInterpolator(list(zip(x, y, z)), point_cloud_values.reshape(-1))
    vals_mesh = interp(X_mesh, Y_mesh, Z_mesh)

# DEPRECATED.
def get_axisparallel_slice(points, values, axis, altitude, n_grid, 
        surface_points, fill_offset=1, fill_value=-1e6, offset=60):
    """ 

    Parameters
    ----------
    fill_offset: float, default=1
        How much above the surface should we start to fill with 
        fill_value.
    fill_value: float, default=-1e6
        Value to use for points outside the region.

    Returns
    -------
    pts_mesh, vals_mesh

    """
    # Create an artificial surface above the real one to make 
    # sure that values outside get interpolated to the fill value.
    artificial_surface = surface_points
    artificial_surface[:, 2] = artificial_surface[:, 2] + fill_offset
    artificial_surface_vals = np.full(artificial_surface.shape[0], fill_value)

    # Add artificial surface to the real data.
    points = np.concatenate((points, artificial_surface), axis=0)
    values = np.concatenate(
            (values.reshape(-1), artificial_surface_vals), axis=0)

    # Get the base dimensions.
    mask = np.ones(points.shape[1], dtype=bool)
    mask[axis] = False # Remove dimension.
    masked_points = points[..., mask]

    # 2D grid for interpolation
    padding = 50 # Add some padding cells outside the volcano, so we make sure these get interpolated to 0.
    pts_1 = np.linspace(masked_points[:, 0].min() - padding, masked_points[:, 0].max() + padding, num=n_grid)
    pts_2 = np.linspace(masked_points[:, 1].min() - padding, masked_points[:, 1].max() + padding, num=n_grid)
    pts_mesh_1, pts_mesh_2 = np.meshgrid(pts_1, pts_2)

    # Compute the constant altitude slice.
    pts_mesh_alt = altitude * np.ones(pts_mesh_1.shape)

    # Combine all points.
    pts_mesh = [pts_mesh_1, pts_mesh_2]
    pts_mesh[axis:axis] = [pts_mesh_alt] # Insert at correct place in the list.
    pts_mesh = np.stack(pts_mesh, axis=2)

    # Only keep data close to the slice to speed-up interpolation.
    ids = np.logical_and(points[:, axis] < altitude + offset, points[:, axis] > altitude - offset)
    interp = LinearNDInterpolator(points[ids], values[ids].reshape(-1))
    
    # Interpolate on the 2D mesh.
    vals_mesh = interp(pts_mesh)

    return pts_mesh, vals_mesh

def get_plotly_surface(x_mesh, y_mesh,z_mesh, vals_mesh):
    """ Builds the plotly surface from meshed data.
    
    Parameters
    ----------
    x_mesh: array (m, n)
    y_mesh: array (m, n)
    z_mesh: array (m, n)
    vals_mesh: array (m, n)
    
    Returns
    -------
    plotly.graph_objects.Surface
    
    """
    return go.Surface(x=x_mesh,
                      y=y_mesh,
                      z=z_mesh,
                      surfacecolor=vals_mesh,
                      coloraxis='coloraxis',
                      # opacityscale=[[0, 0.0], [0.1, 1], [1, 1]]
                      )

def plot_surfaces(slices, title="", cmap='jet', vmin=None, vmax=None):
    """ Plots the surfaces obtained from get_plotly_surface.

    """
    # Find boundaries 
    min_x = min([np.min(slice['x']) for slice in slices])
    min_y = min([np.min(slice['y']) for slice in slices])
    min_z = min([np.min(slice['z']) for slice in slices])
    
    max_x = max([np.max(slice['x']) for slice in slices])
    max_y = max([np.max(slice['y']) for slice in slices])
    max_z = max([np.max(slice['z']) for slice in slices])
    
    # Find color boundaries.
    if vmin is None:
        vmin = np.nanmin([np.nanmin(slice['surfacecolor']) for slice in slices])
    if vmax is None:
        vmax = np.nanmax([np.nanmax(slice['surfacecolor']) for slice in slices])

    fig1 = go.Figure(data=slices)
    fig1.update_layout(
        title_text=title,
        title_x=0.5,
        width=800,
        height=800,
        scene_zaxis_range=[min_z,max_z],
        scene_xaxis_range=[min_x, max_x],
        scene_yaxis_range=[min_y, max_y],
        coloraxis=dict(colorscale=cmap, 
                       colorbar_thickness=25,
                       colorbar_len=0.75,
                       cmin=vmin, cmax=vmax))
    return fig1

def mesh_to_vtkStructuredGrid(X_mesh, Y_mesh, Z_mesh, vals_mesh):
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

def get_standard_slices(grid, values, std_z=15, std_y=50, std_x=50):
    """ Helper function to plot the x-y-z slices of the volcano at standard depth.

    Parameters
    ----------
    grid
    values
    std_z: int, Defaults to 15.
        Altitude (given by index in the mesh) at which to plot the z-slice.
    std_y: int, Defaults to 15.
        Altitude (given by index in the mesh) at which to plot the y-slice.
    std_x: int, Defaults to 15.
        Altitude (given by index in the mesh) at which to plot the x-slice.

    Returns
    -------
    slice_z, slice_y, slice_x
        Slices that can then by plotted using plotly.graph_objects add_trace.

    """
    values_mesh = grid.mesh_values(values)
    
    # z slice.
    X_mesh_slice, Y_mesh_slice, Z_mesh_slice = grid.X_mesh[:, :, std_z], grid.Y_mesh[:, :, std_z], grid.Z_mesh[:, :, std_z]
    values_slice = values_mesh[:, :, std_z]
    slice_z = get_plotly_surface(X_mesh_slice, Y_mesh_slice, Z_mesh_slice, values_slice)

    # y slice.
    X_mesh_slice, Y_mesh_slice, Z_mesh_slice = grid.X_mesh[:, std_y, :], grid.Y_mesh[:, std_y, :], grid.Z_mesh[:, std_y, :]
    values_slice = values_mesh[:, std_y, :]
    slice_y = get_plotly_surface(X_mesh_slice, Y_mesh_slice, Z_mesh_slice, values_slice)

    # x slice.
    X_mesh_slice, Y_mesh_slice, Z_mesh_slice = grid.X_mesh[std_x, :, :], grid.Y_mesh[std_x, :, :], grid.Z_mesh[std_x, :, :]
    values_slice = values_mesh[std_x, :, :]
    slice_x = get_plotly_surface(X_mesh_slice, Y_mesh_slice, Z_mesh_slice, values_slice)
    
    return slice_z, slice_y, slice_x
