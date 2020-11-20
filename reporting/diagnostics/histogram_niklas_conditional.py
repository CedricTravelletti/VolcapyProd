""" Plot an histogram of the excursion volume for the ground truths, i.e.
realisations conditional on Niklas data.

"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


data_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018"

post_samples_folder = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/post_samples"
prior_samples_folder = "/home/cedric/PHD/Dev/VolcapySIAM/reporting/sequential_ivr/results_aws/prior_samples"
reskrig_samples_folder = "/home/cedric/PHD/Dev/VolcapySIAM/data/InversionDatas/stromboli_173018/reskrig_samples"


def main():
    THRESHOLD_low = 700.0

    excu_sizes = []
    for i in range(3, 300):
        cond_real = np.load(
                os.path.join(
                        post_samples_folder,
                        "post_sample_{}.npy".format(i)))
        excu_sizes.append(np.sum(cond_real >= THRESHOLD_low))

    excu_sizes_prior = []
    for i in range(3, 300):
        prior_real = np.load(
                os.path.join(
                        prior_samples_folder,
                        "prior_sample_{}.npy".format(i)))
        excu_sizes_prior.append(np.sum(prior_real >= THRESHOLD_low))

    excu_sizes_reskrig = []
    for i in range(200, 400):
        reskrig_real = np.load(
                os.path.join(
                        reskrig_samples_folder,
                        "prior_sample_{}.npy".format(i)))
        excu_sizes_reskrig.append(np.sum(reskrig_real >= THRESHOLD_low))

    plt.hist(excu_sizes, # color="pink",
            alpha=0.6, label="Niklas conditional")
    plt.hist(excu_sizes_prior, # color="pink",
            alpha=0.6, label="Prior")
    plt.hist(excu_sizes_reskrig, # color="pink",
            alpha=0.6, label="Residual kriging prior samples")
    plt.legend()
    plt.savefig("niklas_conditional_histogram.png", bboch_inches="tight",
            dpi=400)
    plt.show()


if __name__ == "__main__":
    main()
