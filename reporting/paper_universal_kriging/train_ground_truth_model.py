""" Train the model (fault line) used to generate the ground truth.

"""

import sys
import os
import numpy as np
from models import load_paper_models
from loading import load_niklas_volcano_data


base_folder = sys.argv[1]
niklas_volcano_data = load_niklas_volcano_data(base_folder)

# Grid of model hyperparameters.
lambda0s = np.linspace(1.0, 3000, 40)
sigma0s = np.linspace(1, 1400, 40)

base_folder = sys.argv[1]
print("1")
model_lazy = load_paper_models(base_folder)["fault line"]
# Preceding call was lazy. Now do instantiation.
print("1.5")
model = model_lazy()
print("2")

results_folder = os.path.join(base_folder, "paper_universal")
os.makedirs(results_folder, exist_ok=True)
out_path = os.path.join(results_folder, "./train_ground_truth.pkl")

print("Pre-loading done.")
model.train_MLE(
    lambda0s,
    sigma0s,
    niklas_volcano_data["y_std"],
    niklas_volcano_data["G"],
    niklas_volcano_data["y"],
    out_path,
)
