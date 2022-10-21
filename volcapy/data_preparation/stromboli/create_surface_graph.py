""" Create the graph describing the connectivity of the data points. 

This graph is to be used for the dynamic programming inquiries about 
otpimal design on the Stromboli volcano.

First Created: 19.10.2022.

"""
import os
import numpy as np
import networkx as nx

data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"


def main():
    # Load static data.
    data_coords = np.load(os.path.join(data_folder,"surface_data_coords.npy"))
    surface_data_coords = np.load(
            os.path.join(data_folder, "surface_data_coords.npy"))
    niklas_data_inds_insurf = np.load(
            os.path.join(data_folder, "niklas_data_inds_insurf.npy"))

    # Create empty graph.
    surface_graph = nx.Graph()

    # Connect all adjacent cells (4 neighbors) on the surface.
    for i, x in enumerate(surface_data_coords):
        print(i)
        for j, y in enumerate(surface_data_coords):
            # Compute vertical and horizontal distance.
            hdist = np.linalg.norm(
                        data_coords[i, :2] - data_coords[j, :2]
                    )
            vdist = data_coords[j, 2] - data_coords[i, 2]
            # Grid has 50m spacing.
            if (hdist > 0.1 and hdist < 51):
                # Add edge. Vertical distance is positive 
                # if one needs to climb to go from a to b.
                surface_graph.add_edge(
                    i, j,
                    hdist=hdist, vdist=vdist,
                    path_attribute='no_trail')
    # Save at the end.
    nx.write_edgelist(surface_graph, 
            os.path.join(data_folder, "stromboli_surface.edgelist"))

if __name__ == "__main__":
    main()
