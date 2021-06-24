import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set()
sns.set_style("white")
plt.rcParams["font.family"] = "Times New Roman"
plot_params = {
        'font.size': 16, 'font.style': 'oblique',
        'axes.labelsize': 'small',
        'axes.titlesize':'small',
        'legend.fontsize': 'small'
        }
plt.rcParams.update(plot_params)


train_results_path = "/home/cedric/PHD/Dev/VolcapySIAM/data/AISTATS_results_v2/FINAL/train/"


df = pd.read_pickle(os.path.join(train_results_path,
        "test_set_results.pkl"))


# Plot nmll.
fig, ax = plt.subplots(figsize=(8,6))
fig.set_size_inches(6, 6)

my_palette = sns.color_palette("RdBu", 6)
my_palette = my_palette[0:2] + [my_palette[-1]]

ax = sns.lineplot('Test set size', 'Test RMSE', hue='kernel', data=df, 
        palette=my_palette)

plt.savefig("test_set_evolution", bbox_inches="tight", pad_inches=0.1, dpi=400)
plt.show()
