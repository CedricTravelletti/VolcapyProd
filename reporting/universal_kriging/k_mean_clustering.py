""" Perform k-means clustering to cluster data points. 
This is then used for folds in cross-validation.

"""
from volcapy.grid.grid_from_dsm import Grid

import numpy as np
import os
import sys
from sklearn.cluster import KMeans

# Now torch in da place.
import torch

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')

data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018/"
results_folder ="/home/cedric/PHD/Dev/VolcapySIAM/reporting/universal_kriging/results/"
# data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
# results_folder = "/storage/homefs/ct19x463/Dev/VolcapyProd/reporting/universal_kriging/results/"
# os.makedirs(results_folder, exist_ok=True)

def main():
    # Load
    G = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_niklas.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(
            grid.cells).float().detach()
    data_coords = np.load(os.path.join(data_folder, "niklas_data_coords.npy"))

    k = 10
    kmeans = KMeans(init="random", n_clusters=k,
            n_init=10, max_iter=300, random_state=42)
    kmeans.fit(data_coords)

    # Plot clusters.
    import matplotlib.pyplot as plt
    plt.scatter(x=data_coords[:, 0], y=data_coords[:, 1], c=kmeans.labels_)
    plt.show()
