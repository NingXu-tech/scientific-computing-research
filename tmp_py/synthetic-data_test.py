#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from causallearn.search.FCMBased.lingam import VARLiNGAM
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


# VARLiNGAM

# In[ ]:


# Set global font and style
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Configuration
csv_dir = "./dataset/"
csv_files = sorted(glob.glob(os.path.join(csv_dir, "synthetic-data-*-v*.csv")))
var_actual_names = ['x', 'y', 'z', 'p', 'q', 'r']
max_lags = 15
threshold = 0.1

for file_path in csv_files:
    """Process a single dataset with VAR-LiNGAM and visualize results.

    Args:
        file_path (str): Path to the CSV dataset.
    """
    file_name = os.path.basename(file_path)
    print(f"\n=== Processing: {file_name} ===")

    # Load CSV data
    data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    num_vars = data.shape[1]
    var_names = var_actual_names[:num_vars]

    # Fit VAR-LiNGAM
    model = VARLiNGAM(lags=max_lags, criterion=None, prune=False, random_state=42)
    model.fit(data)
    adjacency_matrices = model.adjacency_matrices_

    # Plot instantaneous effect matrix (Lag 0)
    """Plot the instantaneous effect matrix (lag 0)."""
    plt.figure(figsize=(4, 4))
    sns.heatmap(adjacency_matrices[0], annot=True, fmt=".2f", cmap="coolwarm", center=0,
                xticklabels=var_names, yticklabels=var_names, cbar=False)
    plt.title(f"{file_name} - Instantaneous (Lag 0)")
    plt.xlabel("Cause (t)")
    plt.ylabel("Effect (t)")
    plt.tight_layout()
    plt.show()

    # Plot lagged matrices (Lag 1 to 15), 1 row = 5 figures
    """Plot lagged causal matrices for lags 1 through max_lags."""
    lagged_mats = adjacency_matrices[1:max_lags+1]
    cols = 5
    rows = (max_lags + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = axes.flatten()

    for idx, mat in enumerate(lagged_mats):
        ax = axes[idx]
        sns.heatmap(mat, ax=ax, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    xticklabels=var_names, yticklabels=var_names, cbar=False)
        ax.set_title(f"Lag {idx + 1}")
        ax.set_xlabel("Cause (t-lag)")
        ax.set_ylabel("Effect (t)")

    # Turn off any extra subplots
    for i in range(len(lagged_mats), len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"{file_name} - Lagged Causal Matrices (Lag 1 to {max_lags})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Print all significant edges including self-loops
    """Print significant causal edges above the specified threshold."""
    print("\nSignificant causal edges (|weight| > {:.2f}):".format(threshold))
    for lag, mat in enumerate(adjacency_matrices[:max_lags + 1]):
        for i in range(num_vars):  # target
            for j in range(num_vars):  # source
                weight = mat[i, j]
                if abs(weight) > threshold:
                    direction = "positive" if weight > 0 else "negative"
                    t_lag = f"(t)" if lag == 0 else f"(t-{lag})"
                    print(f"{var_names[j]}{t_lag} → {var_names[i]}(t): {direction}, strength = {weight:.3f}")


# Transfer Entropy (synthetic-data-01-v1.csv)

# In[ ]:


# ==== Specify the path to a single CSV file ====
"""Specify the dataset file path to be analyzed."""
file_path = "./dataset/synthetic-data-01-v1.csv"
file_name = os.path.basename(file_path)

# ==== Check if file exists ====
"""Verify that the dataset file exists before proceeding."""
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

print(f"\nProcessing dataset: {file_name}")

# ==== Load data ====
"""Load the CSV dataset into a pandas DataFrame and convert it to numpy array.

The array is transposed to match IDTxl's expected input format:
    shape = [variables × time points].
"""
df = pd.read_csv(file_path)
variable_names = df.columns.tolist()
data_array = df.values.T  # IDTxl expects data in shape: variables × time points

# ==== Create Data object ====
"""Create an IDTxl Data object for TE analysis."""
data = Data(data_array, dim_order='ps')

# ==== Define analysis settings ====
"""Define settings for the Transfer Entropy analysis.

Settings include:
    - CMI estimator
    - Source lag range
    - Number of permutations for significance testing
    - Verbosity level
"""
settings = {
    'cmi_estimator': 'JidtKraskovCMI',
    'max_lag_sources': 5,
    'min_lag_sources': 1,
    'verbosity': 1,
    'n_perm_max_stat': 100,
    'n_perm_omnibus': 100,
    'tau_min': 1,
    'tau_max': 1,
}

# ==== Run transfer entropy analysis ====
"""Run a multivariate Transfer Entropy (TE) analysis on the dataset."""
network_analysis = MultivariateTE()
results = network_analysis.analyse_network(settings=settings, data=data)


# Transfer Entropy (synthetic-data-02-v1.csv)

# In[ ]:


# ==== Specify the path to a single CSV file ====
"""Specify the dataset file path to be analyzed."""
file_path = "./dataset/synthetic-data-02-v1.csv"
file_name = os.path.basename(file_path)

# ==== Check if file exists ====
"""Verify that the dataset file exists before proceeding."""
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

print(f"\nProcessing dataset: {file_name}")

# ==== Load data ====
"""Load the CSV dataset into a pandas DataFrame and convert it to numpy array.

The array is transposed to match IDTxl's expected input format:
    shape = [variables × time points].
"""
df = pd.read_csv(file_path)
variable_names = df.columns.tolist()
data_array = df.values.T  # IDTxl expects data in shape: variables × time points

# ==== Create Data object ====
"""Create an IDTxl Data object for TE analysis."""
data = Data(data_array, dim_order='ps')

# ==== Define analysis settings ====
"""Define settings for the Transfer Entropy analysis.

Settings include:
    - CMI estimator
    - Source lag range
    - Number of permutations for significance testing
    - Verbosity level
"""
settings = {
    'cmi_estimator': 'JidtKraskovCMI',
    'max_lag_sources': 3,
    'min_lag_sources': 1,
    'verbosity': 1,
    'n_perm_max_stat': 100,
    'n_perm_omnibus': 100,
    'tau_min': 1,
    'tau_max': 1,
}

# ==== Run transfer entropy analysis ====
"""Run a multivariate Transfer Entropy (TE) analysis on the dataset."""
network_analysis = MultivariateTE()
results = network_analysis.analyse_network(settings=settings, data=data)


# Transfer Entropy (synthetic-data-03-v1.csv)

# In[ ]:


# ==== Specify the path to a single CSV file ====
"""Specify the dataset file path to be analyzed."""
file_path = "./dataset/synthetic-data-03-v1.csv"
file_name = os.path.basename(file_path)

# ==== Check if file exists ====
"""Verify that the dataset file exists before proceeding."""
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

print(f"\nProcessing dataset: {file_name}")

# ==== Load data ====
"""Load the CSV dataset into a pandas DataFrame and convert it to numpy array.

The array is transposed to match IDTxl's expected input format:
    shape = [variables × time points].
"""
df = pd.read_csv(file_path)
variable_names = df.columns.tolist()
data_array = df.values.T  # IDTxl expects data in shape: variables × time points

# ==== Create Data object ====
"""Create an IDTxl Data object for TE analysis."""
data = Data(data_array, dim_order='ps')

# ==== Define analysis settings ====
"""Define settings for the Transfer Entropy analysis.

Settings include:
    - CMI estimator
    - Source lag range
    - Number of permutations for significance testing
    - Verbosity level
"""
settings = {
    'cmi_estimator': 'JidtKraskovCMI',
    'max_lag_sources': 15,
    'min_lag_sources': 1,
    'verbosity': 1,
    'n_perm_max_stat': 100,
    'n_perm_omnibus': 100,
    'tau_min': 1,
    'tau_max': 1,
}

# ==== Run transfer entropy analysis ====
"""Run a multivariate Transfer Entropy (TE) analysis on the dataset."""
network_analysis = MultivariateTE()
results = network_analysis.analyse_network(settings=settings, data=data)


# Transfer Entropy (synthetic-data-04-v1.csv)

# In[ ]:


# ==== Specify the path to a single CSV file ====
"""Specify the dataset file path to be analyzed."""
file_path = "./dataset/synthetic-data-04-v1.csv"
file_name = os.path.basename(file_path)

# ==== Check if file exists ====
"""Verify that the dataset file exists before proceeding."""
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

print(f"\nProcessing dataset: {file_name}")

# ==== Load data ====
"""Load the CSV dataset into a pandas DataFrame and convert it to numpy array.

The array is transposed to match IDTxl's expected input format:
    shape = [variables × time points].
"""
df = pd.read_csv(file_path)
variable_names = df.columns.tolist()
data_array = df.values.T  # IDTxl expects data in shape: variables × time points

# ==== Create Data object ====
"""Create an IDTxl Data object for TE analysis."""
data = Data(data_array, dim_order='ps')

# ==== Define analysis settings ====
"""Define settings for the Transfer Entropy analysis.

Settings include:
    - CMI estimator
    - Source lag range
    - Number of permutations for significance testing
    - Verbosity level
"""
settings = {
    'cmi_estimator': 'JidtKraskovCMI',
    'max_lag_sources': 15,
    'min_lag_sources': 1,
    'verbosity': 1,
    'n_perm_max_stat': 100,
    'n_perm_omnibus': 100,
    'tau_min': 1,
    'tau_max': 1,
}

# ==== Run transfer entropy analysis ====
"""Run a multivariate Transfer Entropy (TE) analysis on the dataset."""
network_analysis = MultivariateTE()
results = network_analysis.analyse_network(settings=settings, data=data)


# Transfer Entropy (synthetic-data-05-v1.csv)

# In[ ]:


# ==== Specify the path to a single CSV file ====
"""Specify the dataset file path to be analyzed."""
file_path = "./dataset/synthetic-data-05-v1.csv"
file_name = os.path.basename(file_path)

# ==== Check if file exists ====
"""Verify that the dataset file exists before proceeding."""
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

print(f"\nProcessing dataset: {file_name}")

# ==== Load data ====
"""Load the CSV dataset into a pandas DataFrame and convert it to numpy array.

The array is transposed to match IDTxl's expected input format:
    shape = [variables × time points].
"""
df = pd.read_csv(file_path)
variable_names = df.columns.tolist()
data_array = df.values.T  # IDTxl expects data in shape: variables × time points

# ==== Create Data object ====
"""Create an IDTxl Data object for TE analysis."""
data = Data(data_array, dim_order='ps')

# ==== Define analysis settings ====
"""Define settings for the Transfer Entropy analysis.

Settings include:
    - CMI estimator
    - Source lag range
    - Number of permutations for significance testing
    - Verbosity level
"""
settings = {
    'cmi_estimator': 'JidtKraskovCMI',
    'max_lag_sources': 5,
    'min_lag_sources': 1,
    'verbosity': 1,
    'n_perm_max_stat': 100,
    'n_perm_omnibus': 100,
    'tau_min': 1,
    'tau_max': 1,
}

# ==== Run transfer entropy analysis ====
"""Run a multivariate Transfer Entropy (TE) analysis on the dataset."""
network_analysis = MultivariateTE()
results = network_analysis.analyse_network(settings=settings, data=data)


# PCMCI

# In[ ]:


# ------------------- Load Data -------------------
"""Load all CSV files from the dataset directory."""
base_dir = "./dataset/"
csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
csv_files.sort()

if not csv_files:
    print(f"No CSV files found in the directory: {base_dir}")
    exit()

tau_max = 15


def run_pcmci_on_segment(segment, variable_names):
    """Run PCMCI on a single time series segment.

    Args:
        segment (numpy.ndarray): Input time series data of shape (time_steps, variables).
        variable_names (list[str]): List of variable names corresponding to the columns in the dataset.

    Returns:
        numpy.ndarray: Causal strength matrix of shape (num_vars, num_vars, tau_max+1),
            where each entry corresponds to the strength of a causal link
            from one variable to another at a given lag.
    """
    dataframe = pp.DataFrame(segment, var_names=variable_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(significance='analytic'), verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=None)  # No PC filtering, test all pairs
    val_matrix = results['val_matrix']  # shape: (num_vars, num_vars, tau_max+1)
    return val_matrix


# Process each CSV file
"""Iterate through all CSV files, run PCMCI analysis, plot heatmaps, and print causal edges."""
for file_name in csv_files:
    print(f"\nProcessing dataset: {file_name}")
    file_path = os.path.join(base_dir, file_name)
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}. Skipping this dataset.")
        continue

    # Load data
    """Read the CSV file into a DataFrame and convert it to a NumPy array."""
    df = pd.read_csv(file_path)
    variable_names = df.columns.tolist()
    data = df.values  # Convert to numpy array
    
    # Run PCMCI on the entire time series
    """Run PCMCI on the dataset and analyze causal structure."""
    try:
        val_matrix = run_pcmci_on_segment(data, variable_names)
        
        # ------------------- Plot heatmap for lag = 0 -------------------
        """Plot causal strength heatmap for lag=0."""
        plt.figure(figsize=(6, 5))
        sns.heatmap(val_matrix[:, :, 0], annot=True, fmt=".3f",
                    cmap="RdBu_r", center=0,
                    xticklabels=variable_names, yticklabels=variable_names)
        plt.title(f"{file_name} - Causal Strength Heatmap (lag = 0)")
        plt.xlabel("Effect")
        plt.ylabel("Cause")
        plt.tight_layout()
        plt.show()

        # ------------------- Plot heatmaps for lag = 1 to lag = 15 -------------------
        """Plot causal strength heatmaps for lags 1 through tau_max in a 3x5 grid."""
        fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 12))
        for lag in range(1, tau_max + 1):
            row = (lag - 1) // 5
            col = (lag - 1) % 5
            ax = axes[row, col]
            sns.heatmap(val_matrix[:, :, lag], annot=True, fmt=".2f",
                        cmap="RdBu_r", center=0,
                        xticklabels=variable_names, yticklabels=variable_names,
                        ax=ax, cbar=False)
            ax.set_title(f"Lag {lag}")
            ax.set_xlabel("Effect")
            ax.set_ylabel("Cause")

        plt.suptitle(f"{file_name} - Causal Strength Heatmaps (lags 1-15)", y=1.02)
        plt.tight_layout()
        plt.show()

        # ------------------- Print significant causal edges -------------------
        """Print causal edges with strength above a predefined threshold."""
        print(f"\nSignificant causal edges for {file_name} (|strength| > 0.01):")
        threshold = 0.01
        num_vars = val_matrix.shape[0]

        for lag in range(tau_max + 1):
            for i in range(num_vars):
                for j in range(num_vars):
                    strength = val_matrix[i, j, lag]
                    if abs(strength) > threshold:
                        cause = variable_names[i]
                        effect = variable_names[j]
                        print(f"{cause} → {effect} (lag={lag}): {strength:.3f}")
                        
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        continue

