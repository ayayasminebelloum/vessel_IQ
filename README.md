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

The methodology follows five main stages, consistent with the report:

### 1. Data Familiarization and Preprocessing

* Load raw time-series data from multiple sensors (engine, navigation, cargo).
* Synchronize timestamps and align signals to uniform sampling intervals.
* Handle missing values using interpolation or statistical imputation.
* Filter out non-operational periods (e.g., vessel idle state).

### 2. Detecting Relationships Between Data Points

* Compute **Pearson** and **Spearman** correlations to reveal linear and monotonic dependencies.
* Apply **cross-correlation functions** to identify time-lagged relationships.
* Use **Principal Component Analysis (PCA)** to detect dominant patterns in multivariate data.

### 3. Statistical Anomaly Detection

* **Single-sensor analysis:** Apply z-score, IQR, and rolling standard deviation to flag outliers or frozen sensors.
* **Multi-sensor analysis:** Monitor deviations in expected correlations; identify when relationships among sensors diverge from normal patterns.

### 4. Machine Learning Extensions

* Implement unsupervised algorithms capable of learning normal behavior without labels:

  * **Isolation Forest** for anomaly scoring.
  * **DBSCAN** for density-based clustering and noise detection.
  * **Autoencoders** to reconstruct normal sensor relationships; reconstruction error serves as anomaly indicator.
* These methods generalize across vessels and support scalable, automated validation.

### 5. Visualization and Reporting

* Generate correlation heatmaps and anomaly overlays on time-series plots.
* Summarize detected anomalies by sensor and voyage period.
* Produce quantitative quality metrics (percentage of valid points, deviation from expected correlations).

---

## Optional Extension: Transformer-Based Approach

Although not implemented in the initial research phase, transformer models can be incorporated to enhance temporal and inter-sensor modeling.

### Motivation

Traditional methods (statistical, clustering, or autoencoder-based) assume relatively simple correlations. Transformers can model **complex, nonlinear, and long-range dependencies** between sensors over time.

### Possible Implementations

* **Temporal Transformer Autoencoder:**
  Train a sequence-to-sequence model on normal time-series segments; use reconstruction error for anomaly detection.
* **Anomaly Transformer (Zhou et al., 2021):**
  Learns association discrepancy between normal and abnormal patterns in multivariate sensor data.
* **Multimodal Sensor Attention:**
  Use attention weights to visualize which sensors contribute most to detected anomalies, providing interpretability.

### Benefits

* Captures long-term and nonlinear dependencies.
* Learns adaptive correlations automatically.
* Can handle variable-length time windows and missing values.

---

## Expected Outcomes

* Correlation maps highlighting stable inter-sensor relationships.
* Anomaly detection framework combining statistical and ML-based validation.
* Quantitative data quality metrics for each vessel dataset.
* Scalable logic foundation for future rule-based or AI-driven monitoring systems.
* Visualization layer for fleet-wide sensor health tracking.

---

## Dependencies

* Python 3.9+
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* tensorflow or pytorch (for autoencoder or transformer models)

Install dependencies using:

```bash
pip install -r requirements.txt
```

---
## Results and Evaluation

* Detected anomalies across propulsion, navigation, and environmental sensors.
* Identified strong correlations such as RPM ↔ Shaft Power ↔ Fuel Flow.
* Developed automated indicators for frozen sensors and inconsistent relationships.
* Demonstrated improved interpretability through correlation-driven reasoning.

---

## Future Work

1. Integrate contextual metadata (voyage phase, weather, cargo state) for contextual anomaly interpretation.
2. Explore transformer-based architectures for long-term dependency modeling.
3. Develop cross-vessel learning mechanisms to generalize data quality metrics.
4. Combine correlation-driven detection with expert-defined operational logic.
5. Implement real-time visualization dashboards for fleet monitoring.


