""" Train hyperparameters.
"""
import os
import torch
import numpy as np
import pandas as pd

import volcapy.covariance.exponential as exponential_kernel
import volcapy.covariance.matern32 as matern32_kernel
import volcapy.covariance.matern52 as matern52_kernel

from volcapy.grid.grid_from_dsm import Grid
from volcapy.update.updatable_covariance import UpdatableGP


data_folder = "/storage/homefs/ct19x463/Data/InversionDatas/stromboli_173018"


def main():
    print("Main")
    # Load static data.
    F = torch.from_numpy(
            np.load(os.path.join(data_folder, "F_corrected_final.npy"))).float().detach()
    grid = Grid.load(os.path.join(data_folder,
                    "grid.pickle"))
    volcano_coords = torch.from_numpy(grid.cells).float().detach()

    data_coords = torch.from_numpy(
            np.load(os.path.join(data_folder,"niklas_data_coords_corrected_final.npy"))).float()
    data_values = torch.from_numpy(
            np.load(os.path.join(data_folder,"niklas_data_obs_corrected_final.npy"))).float()

    # Remove data points that are too close to each other.
    # prob_inds = np.array([92, 109, 116, 142, 143, 199, 235, 294, 295, 400])
    from scipy.spatial import distance_matrix
    dists = distance_matrix(data_coords.numpy(), data_coords.numpy())
    np.fill_diagonal(dists, 100.0)
    prob_inds, _ = np.where(dists < 40.0)

    # prob_inds = np.array([92, 109, 116, 142, 143, 199, 235, 294, 295, 400,
    #     43, 82, 99, 137, 191, 196, 420])
    F = np.delete(F, prob_inds, axis=0)
    data_coords = np.delete(data_coords, prob_inds, axis=0)
    data_values = np.delete(data_values, prob_inds, axis=0)

    # HYPERPARAMETERS
    data_std = 0.1

    sigma0_exp = 308.89
    m0_exp = 535.39
    lambda0_exp = 1925.0

    sigma0_matern32 = 527.84
    m0_matern32 = 549.15
    lambda0_matern32 = 891.66

    sigma0_matern52 = 349.47
    m0_matern52 = 582.43
    lambda0_matern52 = 436.206

    df = pd.DataFrame(columns=['kernel', 'Test set size', 'repetition',
            'Test RMSE'])

    # Loop over hold-out length:
    n_trains = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    n_trains = [250, 300, 350, 400, 450, 500]
    n_trains = [400, 450, 500]
    n_repetitions = 30

    print("Go")
    for n_train in n_trains:
        print("Train: {}".format(n_train))
        for repetition in range(n_repetitions):
            print("Repetition {}.".format(repetition))
            # Create a random shuffle of the data.
            shuffled_inds = list(range(data_values.shape[0]))
            np.random.shuffle(shuffled_inds)
            F_shuffled = F[shuffled_inds, :]
            data_values_shuffled = data_values[shuffled_inds]
        
            # Test/Train split.
            F_train = F_shuffled[:n_train, :]
            data_values_train = data_values_shuffled[:n_train]
            F_test = F_shuffled[n_train:, :]
            data_values_test = data_values_shuffled[n_train:]

            # Re-create the GPs at every loop.
            gp_exp = UpdatableGP(
                    exponential_kernel, lambda0_exp, sigma0_exp, m0_exp, volcano_coords,
                    n_chunks=400)
            gp_matern32 = UpdatableGP(
                    matern32_kernel, lambda0_matern32, sigma0_matern32, m0_matern32, volcano_coords,
                    n_chunks=400)
            gp_matern52 = UpdatableGP(
                    matern52_kernel, lambda0_matern52, sigma0_matern52, m0_matern52, volcano_coords,
                    n_chunks=400)
            gps = [gp_exp, gp_matern32, gp_matern52]

            for gp in gps:
                print(gp.covariance.cov_module.KERNEL_FAMILY)
                torch.cuda.empty_cache()
                # Condition on training data.
                gp.update(F_train, data_values_train, data_std)
                m_post_m = gp.mean_vec
                # Predict test data.
                data_values_pred = (F_test @ m_post_m.cpu()).reshape(-1)
                test_rmse = torch.sqrt(
                        torch.mean((data_values_test - data_values_pred)**2))

                # Compute negative log predictive density.
                neg_predictive_log_density = gp.neg_predictive_log_density(
                        data_values_test, F_test, data_std, svd=True)

                df = df.append({'kernel': gp.covariance.cov_module.KERNEL_FAMILY,
                        'Test set size': F.shape[0] - n_train,
                        'repetition': repetition,
                        'Test RMSE': test_rmse.detach().item(),
                        'Test neg_predictive_log_density': neg_predictive_log_density.detach().item()
                        }, ignore_index=True)
        # Save after each train set size.
        df.to_pickle("test_set_results.pkl")


if __name__ == "__main__":
    main()
