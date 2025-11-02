# Correlation-Driven Methods for Improving Vessel Sensor Data Quality

## Overview

This repository accompanies the research project **“Correlation-Driven Methods for Improving Vessel Sensor Data Quality.”**
The project investigates statistical and machine learning approaches to improve the integrity of vessel sensor datasets by identifying relationships among data points and detecting anomalies.

Modern ships generate large volumes of sensor data related to propulsion, fuel, cargo, and environmental conditions. However, such data is often affected by sensor drift, frozen readings, outliers, or missing values. This research proposes a **correlation-driven approach**—leveraging inter-sensor dependencies to assess and enhance data quality—before downstream analytics such as performance monitoring or predictive maintenance are applied.

---

## Research Objectives

1. Identify correlations and dependencies among vessel sensors.
2. Detect and classify data anomalies (frozen sensors, outliers, drift).
3. Evaluate the effectiveness of statistical, rule-based, and machine learning methods.
4. Develop an interpretable, scalable framework for vessel data validation.
5. Prepare datasets for higher-level models in fleet analytics and optimization.

---

## Repository Structure

```
vessel-sensor-quality/
│
├── data/
│   ├── raw/                        # Original raw vessel sensor data 
│   ├── merged_csv/                 # Curated vessel sensor data into csv
│   ├── cleaned_csv/                # Cleaned and synchronized sen
    ├── merged_for_correlation.csv  # Merged cleaned sensor data
│   └── correlations                # Correlation matrices, PCA components, summary statistics

├── notebooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_correlation_analysis.ipynb
│   ├── 3_anomaly_detection_statistical.ipynb
│   ├── 4_anomaly_detection_ml.ipynb
│   └── 5_visualization.ipynb
│
├── src/
│   ├── preprocessing.py        # Data synchronization, missing value handling
│   ├── correlation_utils.py    # Pearson, Spearman, cross-correlation, PCA
│   ├── anomaly_stats.py        # z-score, IQR, moving average, residuals
│   ├── anomaly_ml.py           # Isolation Forest, DBSCAN, Autoencoder
│   ├── transformer_extension.py# (Optional) Transformer-based anomaly detection
│   └── visualization.py        # Heatmaps, time-series plots, dashboards
│
├── results/
│   ├── figures/                # Correlation heatmaps, time-series anomaly plots
    ├── analysis                # 2nd analysis phase; correlation        
│   └── metrics/                # Data quality scores, reconstruction errors
│
├── requirements.txt
├── LICENSE 
└── README.md                 
```

---

## Methodology

The methodology follows five main stages, consistent with the project’s structure and implemented across modular components under `src/` and analysis notebooks in `notebooks/`.

---

### 1. Data Familiarization and Preprocessing

* Load and inspect raw vessel sensor data from multiple subsystems (engine, propulsion, navigation, cargo).  
* Synchronize timestamps across signals to ensure temporal alignment.  
* Interpolate missing data and smooth noisy segments using statistical imputation.  
* Normalize and resample all signals to consistent frequency (e.g., 1-minute intervals).  
* Store cleaned datasets under `data/processed/` for downstream correlation and anomaly analysis.

---

### 2. Correlation and Relationship Analysis

* Compute **Pearson**, **Spearman**, and **Kendall** correlations to capture linear and monotonic dependencies.  
* Generate **heatmaps** and **hierarchical clustering dendrograms** to group highly related sensors.  
* Apply **cross-correlation** and **lagged correlation** to detect lead-lag temporal influences between signals.  
* Perform **rolling correlation** to assess stability of relationships over time.  
* Conduct **Principal Component Analysis (PCA)** and **Mutual Information** analysis to uncover nonlinear patterns.  
* Output correlation matrices, visualizations, and summary statistics to `data/correlations/` and `results/figures/`.

---

### 3. Statistical Anomaly Detection

* Implement single-sensor outlier detection via **Z-score**, **IQR**, and **rolling standard deviation** methods.  
* Extend to multi-sensor anomaly detection using **Correlation Anomaly Index (CAI)** and drift tracking.  
* Use **Granger causality tests** to infer directional relationships and identify unexpected temporal changes.  
* Store outlier results in `data/outliers/` and visualization outputs in `results/figures/`.

---

### 4. Machine Learning-Based Extensions

* Integrate unsupervised ML algorithms for robust, label-free anomaly detection:
  * **Isolation Forest** – anomaly scoring based on feature isolation.  
  * **DBSCAN** – clustering and noise-based detection.  
  * **Autoencoders** – reconstruction of normal sensor patterns, using reconstruction error as anomaly signal.  
* Support scalable cross-vessel validation by standardizing feature sets and correlation-based metrics.  
* Modularized implementations available under `src/anomaly_ml.py`.

---

### 5. Visualization and Reporting

* Use `src/visualization.py` to render:
  * Correlation and PCA heatmaps.  
  * Rolling correlation drift plots.  
  * Sensor anomaly overlays and CHI (Correlation Health Index) charts.  
* Summarize quality scores and detected anomalies in `results/metrics/`.  
* Provide interpretable graphics for engineers and analysts to assess vessel data health.

---

## Optional Extension: Transformer-Based Approach

While not yet implemented in the current phase, transformer models are planned for advanced multivariate and temporal correlation modeling.

### Motivation

Conventional statistical and clustering methods assume fixed, often linear dependencies. Transformer-based models can learn **nonlinear, long-range temporal relationships** among sensors dynamically.

### Future Implementations

* **Temporal Transformer Autoencoder:** sequence-to-sequence model trained on normal patterns; reconstruction error used for anomaly scoring.  
* **Anomaly Transformer (Zhou et al., 2021):** models association discrepancies to detect abnormal time-series patterns.  
* **Attention-Based Multisensor Modeling:** highlight influential sensors contributing to abnormal events, improving interpretability.

---

## Expected Outcomes

* Stable correlation maps and time-varying relationship analysis.  
* Hybrid anomaly detection integrating statistical and ML-based components.  
* Quantitative data quality metrics per sensor and per voyage.  
* Scalable codebase for AI-driven sensor validation and health monitoring.  
* Visualization layer enabling intuitive fleet-wide monitoring.

---

## Dependencies

* Python ≥ 3.9  
* pandas, numpy  
* matplotlib, seaborn  
* scikit-learn  
* scipy, statsmodels  
* tensorflow or pytorch (for autoencoder / transformer models)

Install via:
```bash
pip install -r requirements.txt