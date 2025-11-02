import os
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.cluster.hierarchy import linkage
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx

# ---------------------------------------------------------
# Correlation & Statistical Utilities
# ---------------------------------------------------------

def lagged_corr(a, b, max_lag=60):
    """Compute correlation of a and b over time lags in minutes."""
    lags = range(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        shifted_b = b.shift(lag)
        corr = a.corr(shifted_b)
        corrs.append(corr)
    return pd.Series(corrs, index=lags)


def rolling_correlation_all(CLEAN_DIR, OUTPUT_DIR, window=1440):
    """Compute rolling correlation plots for all sensor pairs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith("_cleaned.csv")]
    sensor_names = [f.replace("_cleaned.csv", "") for f in clean_files]

    for sensor_a, sensor_b in combinations(sensor_names, 2):
        try:
            df_a = pd.read_csv(os.path.join(CLEAN_DIR, f"{sensor_a}_cleaned.csv"), parse_dates=["Timestamp"])
            df_b = pd.read_csv(os.path.join(CLEAN_DIR, f"{sensor_b}_cleaned.csv"), parse_dates=["Timestamp"])
            merged = pd.merge(df_a[["Timestamp", "Value"]],
                              df_b[["Timestamp", "Value"]],
                              on="Timestamp",
                              suffixes=(f"_{sensor_a}", f"_{sensor_b}")).set_index("Timestamp")

            rolling_corr = merged[f"Value_{sensor_a}"].rolling(window).corr(merged[f"Value_{sensor_b}"])
            rolling_corr.to_csv(os.path.join(OUTPUT_DIR, f"rolling_corr_{sensor_a}_{sensor_b}.csv"))

        except Exception as e:
            print(f"Error processing {sensor_a} and {sensor_b}: {e}")

    print("Rolling correlation analysis complete for all pairs.")


def compute_chi(CLEAN_DIR, show_top=15):
    """Compute Correlation Health Index (CHI) for all sensors."""
    files = [f for f in os.listdir(CLEAN_DIR) if f.endswith("_cleaned.csv")]
    data_frames = []

    for f in files:
        df = pd.read_csv(os.path.join(CLEAN_DIR, f), usecols=["Timestamp", "Value"], parse_dates=["Timestamp"])
        df.rename(columns={"Value": f.replace("_cleaned.csv", "")}, inplace=True)
        data_frames.append(df)

    merged = data_frames[0]
    for df in data_frames[1:]:
        merged = pd.merge(merged, df, on="Timestamp", how="outer")

    merged = merged.set_index("Timestamp").interpolate(limit_direction="both")
    corr_matrix = merged.corr()
    chi = corr_matrix.mean(axis=1).sort_values(ascending=False)

    return chi


def compute_granger_all(CLEAN_DIR, maxlag=5, p_threshold=0.05):
    """Compute Granger causality for all sensor pairs."""
    clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith("_cleaned.csv")]
    sensor_names = [f.replace("_cleaned.csv", "") for f in clean_files]
    pairs = []

    for sensor_a, sensor_b in combinations(sensor_names, 2):
        try:
            df_a = pd.read_csv(os.path.join(CLEAN_DIR, f"{sensor_a}_cleaned.csv"), parse_dates=["Timestamp"])
            df_b = pd.read_csv(os.path.join(CLEAN_DIR, f"{sensor_b}_cleaned.csv"), parse_dates=["Timestamp"])
            merged = pd.merge(df_a[["Timestamp", "Value"]], df_b[["Timestamp", "Value"]],
                              on="Timestamp", suffixes=(f"_{sensor_a}", f"_{sensor_b}")).dropna()
            merged = merged.select_dtypes(include=["float64", "int64"])

            result = grangercausalitytests(merged, maxlag=maxlag, verbose=False)
            min_p = min([res[0]["ssr_chi2test"][1] for res in result.values()])
            if min_p < p_threshold:
                pairs.append((sensor_a, sensor_b, min_p))

        except Exception as e:
            print(f"Skipping {sensor_a} and {sensor_b}: {e}")

    return pairs
