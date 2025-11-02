import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

# ---------------------------------------------------------
# Visualization Utilities
# ---------------------------------------------------------

def plot_corr_heatmap(df, method="pearson", title="Sensor Correlation Heatmap"):
    corr = df.corr(method=method)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(title)
    plt.show()


def plot_lagged_corr(corr_series, title, xlabel="Lag (minutes)", ylabel="Correlation"):
    corr_series.plot(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.axvline(0, color="black", linestyle="--")
    plt.show()


def plot_rolling_corr(rolling_corr, title="Rolling Correlation (1-day window)"):
    rolling_corr.plot(title=title)
    plt.show()


def plot_sensor_clusters(corr_matrix):
    link = linkage(1 - corr_matrix, method="average")
    plt.figure(figsize=(14, 6))
    dendrogram(link, labels=corr_matrix.columns, leaf_rotation=90)
    plt.title("Sensor Clustering by Correlation")
    plt.show()


def plot_mutual_info(mi_matrix):
    sns.heatmap(mi_matrix, cmap="YlGnBu")
    plt.title("Mutual Information Between Sensors")
    plt.show()


def plot_cai(deviation):
    deviation.plot(title="Correlation Anomaly Index (CAI)")
    plt.show()


def plot_sensor_influence(pairs):
    if not pairs:
        print("No significant causality pairs found.")
        return

    G = nx.DiGraph()
    for src, tgt, p_value in pairs:
        G.add_edge(src, tgt, weight=1 - p_value)

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, with_labels=True,
            node_color="lightblue", node_size=1800,
            font_size=9, edge_color="gray", arrowsize=15)
    plt.title("Sensor Influence Network (Granger Causality)")
    plt.show()


def plot_chi(chi, show_top=15):
    plt.figure(figsize=(12, 5))
    chi.head(show_top).plot(kind="bar", color="steelblue")
    plt.title(f"Correlation Health Index (Top {show_top} Sensors)")
    plt.ylabel("Average Correlation with Other Sensors")
    plt.xlabel("Sensor")
    plt.tight_layout()
    plt.show()
