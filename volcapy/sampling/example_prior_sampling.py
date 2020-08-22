import os
import torch
import numpy as np
from volcapy.grid.grid_from_dsm import Grid
from volcapy.plotting.vtkutils import irregular_array_to_point_cloud
from rpy2.robjects import numpy2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr


def main():
    # Load volcano geometry.
    data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/variance_extraction_benchmark/stromboli_51023_cells"
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = grid.cells

    # Activate auto-wrapping of numpy arrays as rpy2 objects.
    numpy2ri.activate()

    # Import the R RandomFields library.
    rflib = importr("RandomFields")

    # Create the model and sample.
    model = rflib.RMexp(var=3, scale=5)
    simu = rflib.RFsimulate(model, volcano_coords)

    # Back to numpy.
    sample = np.asarray(simu.slots['data']).astype(np.float32)

    irregular_array_to_point_cloud(volcano_coords, sample, "sample.vtk",
            fill_nan_val=-20000.0)


if __name__ == "__main__":
    main()
