""" (manually) Creates the graph describing the tracks on the Stromboli.

This is an interactive script that should be used jointly with the output from 
data_path labelling.py.

The idea is to look at the GIF generated by data_path_labelling and to manually
connect the points.

Here the graph is weighted by the distance between the nodes.

"""
import os
import numpy as np
import networkx as nx


data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018/"

def main():
    # Load
    data_coords = np.load(os.path.join(data_folder,"niklas_data_coords_corrected_final.npy"))

    # Check if exist, otherwise create empty graph.
    if os.path.exists(os.path.join(data_folder, "stromboli_trails.edgelist")):
        ans = input("Graph already exists, would you like to load and modify? (y or n): ")
        if str(ans) == 'y':
            print("Loading existing graph.")
            G = nx.read_edgelist(os.path.join(data_folder, "stromboli_trails.edgelist"))
            print(list(G.edges()))
        else: G = nx.DiGraph()
    else: G = nx.DiGraph()

    def connect_points(ind1, ind2):
        """ Connects two datapoints with an edge. Vertical distance 
        is positive if have to climb to go from A to B.

        By default connect in both directions.
    
        """
        # Compute vertical and horizontal distance.
        hdist = np.linalg.norm(
                    data_coords[ind1, :2] - data_coords[ind2, :2]
                )
        vdist = data_coords[ind2, 2] - data_coords[ind1, 2]

        G.add_edge(
                ind1, ind2,
                hdist=hdist, vdist=vdist,
                path_attribute='trail')
        G.add_edge(
                ind2, ind1,
                hdist=hdist, vdist=-vdist,
                path_attribute='trail')

    i = 0
    while i < data_coords.shape[0]:
        # Save periodically.
        if i % 20 == 0:
            nx.write_edgelist(G, os.path.join(data_folder, "stromboli_trails.edgelist"))

        print("Node {}".format(i))
        while True:
            ind_to_connect = input(
                    "Connect cell {} to cell (s to skip to next cell, n to connect next and skip, w to save): ".format(i))
            try:
                ind_to_connect = int(ind_to_connect)
                print("Connecting {} and {}.".format(i, ind_to_connect))
                connect_points(i, ind_to_connect)
            except:
                if ind_to_connect == "n":
                    print("connecting to next and skipping")
                    connect_points(i, i + 1)
                    i = i + 1
                elif ind_to_connect == "s":
                    print("Skipping.")
                    i = i + 1
                elif ind_to_connect == "j":
                    ind_to_jump = input("Type index of node you would like to jump to: ")
                    print("Jumping to node {}.".format(ind_to_jump))
                    i = int(ind_to_jump)
                elif ind_to_connect == 'w':
                    print("Saving")
                    nx.write_edgelist(G, os.path.join(data_folder, "stromboli_trails.edgelist"))
                else: 
                    print("Input has to be integer.")

    print("Done. Saving.")
    nx.write_edgelist(G, os.path.join(data_folder, "stromboli_trails.edgelist"))

if __name__ == "__main__":
    main()