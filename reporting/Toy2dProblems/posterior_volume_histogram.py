import numpy as np
import matplotlib.pyplot as plt


samples = np.load("./results_histo/samples.npy")
n_ground_truths = samples.shape[1]
n_samples = samples.shape[2]

# For each ground truth, plot volume histogram.
fig, axs = plt.subplots(3, 5)

prior_vol_exp = 0
post_vol_exp = 0

for i in range(n_ground_truths):
    ground_truth = np.load("./results_histo/ground_truth_{}.npy".format(i))

    # Compute sizes at the end.
    sizes = []
    for j in range(n_samples):
        sizes.append(np.sum(samples[:, i, j] > 3))
        post_vol_exp += np.sum(samples[:, i, j] > 3)
    ax = fig.axes[i]
    ax.hist(sizes, alpha=0.4)
    ax.axvline(np.sum(ground_truth > 3))
    
    prior_vol_exp += np.sum(ground_truth > 3)

print("Prior volume expectation {}".format(prior_vol_exp/n_ground_truths))
print("Posterior volume expectation {}".format(post_vol_exp/(n_samples * n_ground_truths)))

plt.show()
