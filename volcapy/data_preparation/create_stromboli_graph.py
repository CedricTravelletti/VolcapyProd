""" (manually) Creates the graph describing the tracks on the Stromboli.

"""
import os
import numpy as np
import networkx as nx
import seaborn
import matplotlib.pyplot as plt


output_path = "/home/cedric/PHD/Dev/VolcapySIAM/volcapy/data_preparation/data_path_labelling"
data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/inversion_data_dsm_coarse"

def main():
    # Load
    data_coords = np.load(os.path.join(data_folder,"niklas_data_coords.npy"))

    # Create empty graph.
    G = nx.Graph()

    def connect_points(ind1, ind2):
        """ Connects two datapoints with an edge. Edge weight will be given by
        euclidean distance between the two points.
    
        """
        G.add_edge(
                ind1, ind2,
                weight=np.linalg.norm(data_coords[ind1, :] - data_coords[ind2, :]))

    # for i in range(data_coords.shape[0]):
    for i in range(10):
        # Save periodically.
        if i % 20 == 0:
            nx.write_edgelist(G, "stromboli.edgelist")

        print("Node {}".format(i))
        while True:
            ind_to_connect = input(
                    "Connect cell {} to cell (-1 to skip to next cell): ".format(i))
            try:
                ind_to_connect = int(ind_to_connect)

                if ind_to_connect < 0:
                    print("Skipping.")
                    break
                else:
                    print("Connecting {} and {}.".format(i, ind_to_connect))
                    connect_points(i, ind_to_connect)
            except:
                if ind_to_connect == "n":
                    print("connecting to next and skipping")
                    connect_points(i, i + 1)
                    break
                else: 
                    print("Input has to be integer.")

    print("Done. Saving.")
    nx.write_edgelist(G, "stromboli.edgelist")

if __name__ == "__main__":
    main()
