""" Train a constant GP on the log-gaussian ground truth used 
in the comparison of universal and traditional inversion.

"""
from volcapy.inverse.inverse_gaussian_process import InverseGaussianProcess
import volcapy.covariance.matern52 as kernel
from volcapy.grid.grid_from_dsm import Grid

import numpy as np
import os
import sys

# Now torch in da place.
import torch

# General torch settings and devices.
torch.set_num_threads(8)
gpu = torch.device('cuda:0')
cpu = torch.device('cpu')

data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"
results_folder = "/storage/homefs/ct19x463/Dev/VolcapyProd/reporting/universal_kriging/results/const_vs_univ_loggaussian/"
os.makedirs(results_folder, exist_ok=True)


def main():
    # Load
    G = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_niklas.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(
            grid.cells).float().detach()

    # Define GP model.
    data_std = 0.1
    sigma0 = 1.0
    m0 = 2139.1
    lambda0 = 200.0

    ground_truth = torch.from_numpy(np.load(os.path.join(results_folder, "ground_truth.npy")))
    synth_data = torch.from_numpy(np.load(os.path.join(results_folder, "synth_data.npy")))

    # Now train GP model on it.
    myGP = InverseGaussianProcess(m0, sigma0, lambda0,
            volcano_coords, kernel)
    myGP.train(np.linspace(1.0, 2000, 20), G, synth_data, data_std,
            out_path=os.path.join(results_folder, "./train_res_matern52.pck"),
            n_epochs=2000, lr=0.5,
            n_chunks=80,
            n_flush=1)


if __name__ == "__main__":
    main()
