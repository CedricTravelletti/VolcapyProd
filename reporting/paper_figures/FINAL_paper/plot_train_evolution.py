import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set()
sns.set_style("white")
plt.rcParams["font.family"] = "Times New Roman"
plot_params = {
        'font.size': 20, 'font.style': 'oblique',
        'axes.labelsize': 'small',
        'axes.titlesize':'small',
        'legend.fontsize': 'small',
        'legend.title_fontsize': 'x-small'
        }
plt.rcParams.update(plot_params)


train_results_path = "/home/cedric/PHD/Dev/VolcapySIAM/data/AISTATS_results_v2/FINAL/train/"


df_matern52 = pd.read_pickle(os.path.join(train_results_path,
        "train_res_matern52_final.pck"))
df_matern52['kernel'] = 'Matern 5/2'

df_matern32 = pd.read_pickle(os.path.join(train_results_path,
        "train_res_matern32_merged2.pck"))
df_matern32['kernel'] = 'Matern 3/2'

df_exponential = pd.read_pickle(os.path.join(train_results_path,
        "train_res_exponential_merged2.pck"))
df_exponential['kernel'] = 'Exponential'

# Compute practical range for each.
from volcapy.covariance.utils import compute_practical_range
import volcapy.covariance.exponential as exponential
import volcapy.covariance.matern32 as matern32
import volcapy.covariance.matern52 as matern52

df_exponential['practical range'] = df_exponential['lambda0'].map(
        lambda a: compute_practical_range(exponential, a))
df_matern32['practical range'] = df_matern32['lambda0'].map(
        lambda a: compute_practical_range(exponential, a))
df_matern52['practical range'] = df_matern52['lambda0'].map(
        lambda a: compute_practical_range(exponential, a))

df = pd.concat([df_matern52, df_matern32,
        df_exponential]).reset_index(drop=True)

# Print MLE for each kernel.
print("MLE exponential:")
print(df_exponential[df_exponential.nll == df_exponential.nll.min()])
print("MLE Matern 3/2:")
print(df_matern32[df_matern32.nll == df_matern32.nll.min()])
print("MLE Matern 5/2:")
print(df_matern52[df_matern52.nll == df_matern52.nll.min()])


# Plot nmll.
fig, ax = plt.subplots(figsize=(8,6))
fig.set_size_inches(6, 6)

my_palette = sns.color_palette("RdBu", 6)
my_palette = my_palette[0:2] + [my_palette[-1]]

ax = sns.lineplot('practical range', 'nll', ci=None, hue='kernel', data=df, 
        palette=my_palette)

ax.set_xlabel("practical range [m]")
ax.set_ylabel("Marginal Negative Log-likelihood")
ax.set_xlim([0, 4500])
plt.savefig("train_evolution_nll", bbox_inches="tight", pad_inches=0.1, dpi=400)
plt.show()

# Plot hyperparams.
fig, ax = plt.subplots(figsize=(8,6))
fig.set_size_inches(6, 6)

my_palette = sns.color_palette("RdBu", 6)
my_palette = my_palette[0:2] + [my_palette[-1]]


df_melted = pd.melt(df, id_vars=['practical range', 'kernel'], value_vars=['m0', 'sigma0'],
        ignore_index=True, 
        var_name='parameter', value_name='parameter value')

ax = sns.lineplot('practical range', 'parameter value', ci=None, hue='kernel', style="parameter",
        data=df_melted, palette=my_palette)

ax.set_xlabel("practical range [m]")
ax.set_ylabel("Optimal parameter value [kg/m3]")
ax.set_xlim([0, 4500])
plt.savefig("train_evolution_hyperparams", bbox_inches="tight", pad_inches=0.1, dpi=400)
plt.show()

# Plot train RMSE.
fig, ax = plt.subplots(figsize=(8,6))
fig.set_size_inches(6, 6)

my_palette = sns.color_palette("RdBu", 6)
my_palette = my_palette[0:2] + [my_palette[-1]]

ax = sns.lineplot('practical range', 'train_RMSE', ci=None, hue='kernel', data=df, 
        palette=my_palette)

ax.set_xlabel("practical range [m]")
ax.set_ylabel("RMSE on train set [mGal]")
ax.set_xlim([0, 4500])
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)

plt.savefig("train_evolution_rmse", bbox_inches="tight", pad_inches=0.1, dpi=400)
plt.show()
