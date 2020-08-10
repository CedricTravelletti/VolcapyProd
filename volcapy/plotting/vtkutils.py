# -*- coding: utf-8 -*-
""" Convert cells coords and measurement coords to vtk for plotting with
paraview.
"""
import vtk
import numpy as np
 
     
class VtkPointCloud:
    """ Class for unstructured point data.

    """
    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)

        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)
 
    def addPoint(self, point, point_data):
        """ Point with data.

        Parameters
        ----------
        point
        point_data: float
            Scalar field value at point. Will be used for coloring. For
            example, provide point[2] to color by height.

        """
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])

            self.vtkValues.InsertNextValue(point_data)
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)

            # Perso
            self.vtkActor.GetProperty().SetPointSize(2)

        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkValues.Modified()
        
        # Perso
        self.vtkActor.Modified()
 
    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkValues = vtk.vtkDoubleArray()
        self.vtkValues.SetName('PointValues')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkValues)
        self.vtkPolyData.GetPointData().SetActiveScalars('PointValues')

class VtkVectorCloud(VtkPointCloud):
    """ Same as point cloud, but with vector data.

    """
    def __init__(self, n_dims, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
        self.n_dims = n_dims
        super(VtkVectorCloud, self).__init__(zMin=-10.0, zMax=10.0, maxNumPoints=1e6)

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkValues = vtk.vtkDoubleArray()

        # Set number of dimensions
        self.vtkValues.SetNumberOfComponents(self.n_dims)

        self.vtkValues.SetName('PointValues')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkValues)
        self.vtkPolyData.GetPointData().SetActiveScalars('PointValues')

    def addPoint(self, point, point_data):
        """ Point with data.

        Parameters
        ----------
        point
        point_data: List
            Vector field value at point.

        """
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])

            self.vtkValues.InsertNextTuple(point_data)

            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)

            # Perso
            self.vtkActor.GetProperty().SetPointSize(2)

        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkValues.Modified()
        
        # Perso
        self.vtkActor.Modified()
 
def txt_to_vtk_pointcloud(path):
    """ Generate a point cloud from a text file.
    File should contain  x y z coordinates in 3 columns. First two lines are
    header.

    Parameters
    ----------
    path: str
        Path to file containing coords.

    """
    data = np.genfromtxt(path, dtype=float, skip_header=2, usecols=[0,1,2])
     
    pointCloud = VtkPointCloud()
    for k in range(np.size(data,0)):
        point = data[k]
        # Add point and color by z value.
        pointCloud.addPoint(point, point[2])
         
    return pointCloud

def save_point_cloud(point_cloud, path):
    """ Save a VTK point cloud to file.

    Parameters
    ----------
    point_cloud
    path: str
        Path to output file. Must have a .vtk extension.

    """
    # Check file extension, since paraview wont work otherwise.
    if not path.endswith(".vtk"):
        raise ValueError("Output file extension must be .vtk .")

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(point_cloud.vtkPolyData)
    writer.Write()

def _array_to_point_cloud(coords, data, output_path):
    """ Convert a list of points, stored in a numpy array to a VTK point cloud.

    Parameters
    ----------
    coords: array
        Coordinates array (n_points, n_dims).
    data: array
        Data array (n_points).
    output_path: str
        Path to output file. Should end with .vtk."

    """
    pointCloud = VtkPointCloud()
    for point, data in zip(coords, data):
        pointCloud.addPoint(point, data)
         
    save_point_cloud(pointCloud, output_path)

def irregular_array_to_point_cloud(coords, data, output_path, fill_nan_val=0.0):
    """ Same as array to point cloud, but takes care of putting back in a prism
    and filling missing values with zero. This tends to make visualization
    easier.

    Parameters
    ----------
    coords: array
        Coordinates array (n_points, n_dims).
    data: array
        Data array (n_points).
    output_path: str
        Path to output file. Should end with .vtk."
    fill_nan_val: float
        Value with which to fill the missing values. Defaults to 0.0.

    """
    # Reshape if necessary.
    if len(data.shape) < 2:
        data = data[:, None]

    # Try to fit in a regular grid by filling non-volcano points with zeros.
    xy_coords = np.unique(coords[:, :2], axis=0)
    z_levels = np.unique(coords[:, 2])

    # Build the 3D grid (inelegant, but fuck it).
    regular_points = np.empty((0, 3), dtype=np.float32)
    for level in z_levels:
        regular_points = np.vstack([regular_points, np.hstack([xy_coords, np.repeat(level,
                xy_coords.shape[0])[:, None]])])
    # Now only keep points in regular grid that are not in volcano.
    non_grid_coords = multidim_setdiff(regular_points, coords)

    # Set their density to zero and stack.
    data_non_grid = np.full((non_grid_coords.shape[0], 1),
            fill_nan_val, dtype=np.float32)

    # Concatenate and save.
    all_coords = np.vstack([coords, non_grid_coords])
    all_data = np.vstack([data, data_non_grid])

    _array_to_point_cloud(all_coords, all_data, output_path)

def multidim_setdiff(arr1, arr2):
    """ Returns elements of arr1 that are not in arr2.

    """
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    intersected = np.setdiff1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

def array_to_vector_cloud(coords, data, output_path):
    """ Convert a list of points, stored in a numpy array to a VTK point cloud.

    Parameters
    ----------
    coords: array
        Coordinates array (n_points, n_dims).
    data: array
        Data array (n_points, d).
    output_path: str
        Path to output file. Should end with .vtk."

    """
    n_dims = data.shape[1]
    print(n_dims)
    pointCloud = VtkVectorCloud(n_dims)
    for i, point in enumerate(coords):
        pointCloud.addPoint(point, data[i, :])
         
    save_point_cloud(pointCloud, output_path)
