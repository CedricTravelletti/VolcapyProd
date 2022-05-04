import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set()
sns.set_style("white")
plt.rcParams["font.family"] = "serif"
plot_params = {
        'font.size': 18, 'font.style': 'oblique',
        'axes.labelsize': 'x-small',
        'axes.titlesize':'x-small',
        'legend.fontsize': 'x-small',
        'xtick.labelsize': 'x-small',
        'ytick.labelsize': 'x-small'
        }
plt.rcParams.update(plot_params)


train_results_path = "/home/cedric/PHD/Dev/VolcapySIAM/data/AISTATS_results_v2/FINAL/train/"


# df = pd.read_pickle(os.path.join(train_results_path,
#         "test_set_results.pkl"))
df = pd.read_pickle("./test_set_results.pkl")

df['Train set size'] = 501 - df['Test set size']


# Plot nmll.
fig = plt.figure(figsize=(5, 5), dpi=400)

my_palette = sns.color_palette("RdBu", 6)
my_palette = my_palette[0:2] + [my_palette[-1]]

ax = sns.lineplot('Train set size', 'Test RMSE', hue='kernel', data=df, 
        palette=my_palette)

plt.savefig("test_set_evolution_rmse", bbox_inches="tight", pad_inches=0.1, dpi=400)
plt.show()

fig = plt.figure(figsize=(5, 5), dpi=400)
ax = sns.lineplot('Train set size', 'Test neg_predictive_log_density', hue='kernel', data=df, 
        palette=my_palette)
ax.set_ylabel("Test negative log-predictive density")


plt.savefig("test_set_evolution_neg_log_dens", bbox_inches="tight", pad_inches=0.1, dpi=400)
plt.show()
