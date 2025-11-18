# Correlation-Driven Methods for Improving Vessel Sensor Data Quality

## Overview

This repository contains the full workflow for **Correlation-Driven Methods for Improving Vessel Sensor Data Quality**, a research and engineering project focused on improving the integrity, reliability, and interpretability of vessel sensor data. Modern maritime systems generate large volumes of telemetry from propulsion, navigation, environmental, and mechanical subsystems — but real-world data is often noisy, incomplete, unsynchronized, or corrupted.

This project implements a **multi-stage data quality pipeline** built around correlation-driven insights, statistical diagnostics, and machine learning models. It aligns with Kongsberg's goals of improving fleet analytics, anomaly detection, and vessel health monitoring.

---

## Objectives

The project is structured around six major goals:

1. **Data ingestion & synchronization** across multiple sensor subsystems.
2. **Exploratory analysis** to understand sensor behaviors and detect structural issues.
3. **Correlation analysis** to map sensor dependencies.
4. **Statistical diagnostics** to detect outliers, frozen sensors, drift, and noise.
5. **Machine learning integration** for anomaly detection and validation.
6. **Model testing & evaluation** using engineered datasets and validation loops.

Each objective corresponds to a notebook under `/notebooks` and modular code under `/src`.

---

## Repository Structure

```
KONGSBERG/
│
├── data/
│   ├── cleaned_csv/            # Fully cleaned per-sensor datasets
│   ├── derived/                # Engineered features and derived metrics
│   ├── diagnostics/            # Statistical test outputs
│   ├── heartbeat/              # Heartbeat-level system status data
│   ├── mapping/                # Sensor mapping tables
│   ├── ml_models/              # Saved ML models (IForest, AE, etc.)
│   ├── network/                # Network topology or subsystem grouping
│   ├── raw/                    # Raw sensor CSVs
│   ├── testing/                # Validation datasets
│   ├── merged_for_correlation.csv
│   └── merged_sensor_data.csv
│
├── images/                    # Figures exported from notebooks
│
├── notebooks/
│   ├── correlation_analysis.ipynb
│   ├── data_exploration.ipynb
│   ├── ml_integration.ipynb
│   ├── outlier_verification.ipynb
│   ├── statistical_diagnostics.ipynb
│   └── testing_models.ipynb
│
├── results/                   # Output metrics, diagnostics, reports
│
├── src/
│   ├── preprocessing.py
│   ├── correlation_utils.py
│   ├── statistical_tools.py
│   ├── ml_integration.py
│   ├── model_testing.py
│   └── utils.py
│
├── venv/
├── LICENSE
├── requirements.txt
└── README.md (this file)
```

---

## Workflow Breakdown (Aligned With the 6 Notebooks)

For the full processing pipeline, see the workflow diagram here:  
[**Flowchart**](https://www.figma.com/board/QrxNQJUzQ1XQZcP7l9qmqv/Kongsberg-Sensor-Health-System---Updated-Flowchart?node-id=0-1&t=2zkH9O8ZTWRPtbni-1)

---

## 1. Data Exploration (`data_exploration.ipynb`)

**Goal:** Understand the raw sensor structure, sampling rates, missing values, and anomalies.

### Key tasks

* Load raw CSVs from `/data/raw`
* Standardize column names
* Detect timestamp issues
* Plot primary signals & density curves
* Identify feature groups
* Compute basic statistics (mean, std, count)

**Outputs:**

* `/results/plots/eda_density_fixed.png`
* Summary of missing data
* Initial signal quality flags

---

## 2. Correlation Analysis (`correlation_analysis.ipynb`)

**Goal:** Build a correlation-driven map of all sensor dependencies.

### Methods used

* Pearson, Spearman correlation matrices
* Rolling window correlations
* Lag correlation computation
* PCA-based dimensionality reduction
* Visualization of clustering between sensors

**Inputs:** `merged_for_correlation.csv`

**Outputs:**

* Heatmaps (saved to `/results`)
* PCA components
* Sensor grouping & hierarchy information

---

## 3. Statistical Diagnostics (`statistical_diagnostics.ipynb`)

**Goal:** Use classical statistical methods to detect structural anomalies.

### Techniques implemented

* Z-score & modified Z-score
* IQR + Tukey fences
* Rolling mean/variance deviation
* Drift detection
* Residual-based anomaly scoring

**Outputs:**

* Diagnostics tables → `/data/diagnostics`
* Time-series plots with highlighted anomalies

---

## 4. Outlier Verification (`outlier_verification.ipynb`)

**Goal:** Validate anomalies detected earlier using visual inspection and rule-based checks.

### Includes

* Manual sensor comparison
* Multi-sensor consistency checks
* Rule-based thresholds from SME expertise
* Cross-verifying outliers with subsystem logic

**Outputs:** in `/diagnostics/outlier_verification`.

---

## 5. Machine Learning Integration (`ml_integration.ipynb`)

**Goal:** Integrate ML-based anomaly detectors using cleaned & engineered features.

### ML models used

* **Isolation Forest** (unsupervised)
* **Local Outlier Factor**
* **Autoencoder reconstruction error**
* **Ensemble scoring** combining statistical + ML methods

### Key steps

* Feature scaling
* Train-test split from `/data/testing`
* Model calibration using domain-specific constraints

**Outputs:**

* Saved models → `/data/ml_models/checkpoints`
* ML anomaly scores → `/data/ml_models`

---

## 6. Testing & Validation (`testing_models.ipynb`)

**Goal:** Validate ML and statistical models on unseen datasets.

### Validation includes

* Precision–recall for anomaly flags
* Drift detection generalization across voyages
* Comparison between anomaly detection pipelines
* Reconstruction error evaluation

**Outputs:**

* Model test metrics
* Final performance charts
* Candidate model selection for deployment

---

## Methodology Summary

Across all stages, the project performs:

* **Timestamp normalization**
* **Missing-value imputation**
* **Cross-sensor correlation mapping**
* **Unsupervised anomaly detection**
* **Validation on multiple datasets**
* **Model persistence & reproducibility**

---

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Python ≥ 3.9 recommended.

---

## License

MIT License.

