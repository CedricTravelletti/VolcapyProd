""" Produce plots for manual labelling of data sites.
We want to group data points according to paths.
This script sequentially plots the data points so that we can identify which
indices belong to which path.

"""
import os
import numpy as np
import seaborn
import matplotlib.pyplot as plt


output_path = "/home/cedric/PHD/Dev/VolcapySIAM/volcapy/data_preparation/data_path_labelling"
data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/inversion_data_dsm_coarse"

def main():
    # Load
    data_coords = np.load(os.path.join(data_folder,"niklas_data_coords.npy"))

    # Plot full set.
    fig, ax = plt.subplots()
    ax.scatter(data_coords[:, 0], data_coords[:, 1], s=0.2, c=data_coords[:, 2])

    # Add annotations.
    for i in range(data_coords.shape[0]):
        ax.annotate(str(i), (data_coords[i, 0], data_coords[i, 1]), size=1.5)
    plt.savefig(os.path.join(output_path, "full_dataset.png"), dpi=800)

    for i, current_pt in enumerate(data_coords):
        plt.figure()
        # Plot full set.
        plt.scatter(data_coords[:, 0], data_coords[:, 1], s=10, c="black")
        # Plot cells processed so far.
        plt.scatter(data_coords[:i, 0], data_coords[:i, 1], s=20, c="lightgreen")
        # Plot current one.
        plt.scatter(data_coords[i, 0], data_coords[i, 1], s=80, c="red")
        plt.title("Cell {} / {}".format(i, data_coords.shape[0]))
        plt.savefig(os.path.join(output_path, "fig_{}.png".format(i)))

if __name__ == "__main__":
    main()
