# Correlation-Driven Methods for Improving Vessel Sensor Data Quality
### A Tiered Graph Model for Maritime Sensor Health Monitoring

**Author:** Aya-Yasmine Belloum · IE University · BCSAI · 2025–2026  
**Supervisors:** Prof. Daniel Precioso Garcelán · Prof. David Gómez Ullate Oteiza  
**Industry Partner:** Kongsberg Maritime 

---

## Overview

This repository contains the full implementation of the **Tiered Graph Model (TGM)** — a hybrid, correlation-driven framework for monitoring and improving the data quality of industrial sensor networks aboard LNG carriers.

Applied to real operational data from a live vessel connected to Kongsberg Maritime's PI System, the TGM processes **28,196,293 training records** and **5,729,769 test records** across 45 sensors and 9 cargo subsystems, producing a unified per-sensor **Hybrid Health Score (HHS)** on a 0–100 reliability scale.

The central finding: all 45 sensors pass conventional heartbeat checks and report 99.98% valid `ValueStatus` — yet the TGM identifies structurally unstable subsystems, detects a multivariate anomaly cluster invisible to individual-sensor monitoring, and provides interpretable diagnostics that maintenance engineers can act on directly.

---

## Key Results

| Metric | Value |
|--------|-------|
| Training records | 28,196,293 |
| Test records | 5,729,769 |
| Sensors monitored | 45 (across 9 subsystems) |
| All sensors heartbeat status | Alive (100%) |
| ValueStatus validity | 99.98% good — yet masks structural degradation |
| CHI — Cargo temperature subsystems | 0.327–0.335 (Good) |
| CHI — Volume subsystem | −0.193 (Structurally decoupled) |
| CHI — Level subsystem | −0.197 (Structurally decoupled) |
| SRI Tukey upper fence | 118.1 |
| Flagged high-SRI sensors | 5 (all Volume: CT030.13, CT031.13, CT032.13, CT033.13, CT_TOTALVOL) |
| CT_TOTALVOL SRI | 70,139.97 |
| Autoencoder best val_loss | 0.04715 (epoch 1) |
| Autoencoder test mean reconstruction error | 0.049197 |
| Autoencoder 3σ threshold | 0.990968 |
| Anomaly events detected (test set) | 83 |
| GRU test RMSE / MAE | 0.09842 / 0.007765 |
| Isolation Forest outlier ratio | 8.67% |
| High-drift sensors | CT030.13 (2.16%), CT033.13 (2.04%), CT033.12 (2.04%) |

---

## The Tiered Graph Model (TGM)

The TGM processes sensor time-series through three sequential analytical tiers, each enriching the context for the next. All tier outputs are fused into the HHS.

```
┌─────────────────────────────────────────────────────────────┐
│  TIER 1 — Statistical Diagnostics                           │
│  Heartbeat classification · SRI · Dynamic z-score outliers  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  TIER 2 — Correlation Analysis                              │
│  CHI · Granger causality network · Mutual information        │
│  Rolling & time-lagged correlations · Hierarchical clusters  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  TIER 3 — Machine Learning Anomaly Detection                │
│  Denoising Autoencoder · GRU Forecaster · Isolation Forest  │
│  Outlier Neighborhood Verification (ONV)                     │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  HYBRID HEALTH SCORE                                        │
│  HHS = (1 − Norm(0.4×AE + 0.4×GRU + 0.2×SRI)) × 100       │
│  Continuous 0–100 per-sensor reliability index              │
└─────────────────────────────────────────────────────────────┘
```

**Full workflow diagram:** [Kongsberg Sensor Health System — Figma Flowchart](https://www.figma.com/board/QrxNQJUzQ1XQZcP7l9qmqv/Kongsberg-Sensor-Health-System---Updated-Flowchart?node-id=0-1&t=2zkH9O8ZTWRPtbni-1)

---

## Repository Structure

```
vessel_IQ/
│
├── data/                       # However hidden for privacy reasons
│   ├── cleaned_csv/            # Fully cleaned per-sensor datasets
│   ├── derived/                # Engineered features and derived metrics
│   ├── diagnostics/            # Statistical test outputs
│   ├── heartbeat/              # Heartbeat-level system status data
│   ├── mapping/                # Sensor mapping tables
│   ├── ml_models/              # Saved ML models (IForest, AE, GRU, etc.)
│   ├── network/                # Granger causality graphs (GEXF format)
│   ├── raw/                    # Raw sensor CSVs from Kongsberg's PI System
│   ├── testing/                # Held-out test datasets (Sep–Nov 2025)
│   ├── merged_for_correlation.csv
│   └── merged_sensor_data.csv
│
├── images/                     # Figures exported from notebooks
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_correlation_analysis.ipynb
│   ├── 03_statistical_diagnostics.ipynb
│   ├── 04_outlier_verification.ipynb
│   ├── 05_ml_integration.ipynb
│   └── 06_testing_models.ipynb
│
├── results/                    # Output metrics, diagnostics, reports
│
├── src/
│   ├── preprocessing.py        # Ingestion pipeline, resampling, normalisation
│   ├── correlation_utils.py    # CHI, Granger causality, rolling correlations
│   ├── statistical_tools.py    # SRI, heartbeat check, z-score, drift detection
│   ├── ml_integration.py       # Autoencoder, GRU, Isolation Forest, ONV
│   └── model_testing.py        # Evaluation metrics, HHS computation
│    
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Notebook Workflow

### 1. Data Exploration (`01_data_exploration.ipynb`)

**Goal:** Understand raw sensor structure, sampling rates, missing values, and signal characteristics.

Key tasks:
- Load raw CSVs from `/data/raw` (.meta` + `.txt` export format)
- Standardise column names and parse timestamps to UTC-aware datetime
- Detect timestamp gaps, duplicate records, and format inconsistencies
- Plot primary signals and density curves per subsystem
- Compute basic statistics (mean, std, missing rate) across all 45 sensors

**Outputs:**
- `/results/plots/eda_density_fixed.png`
- Summary of missing data rates and initial signal quality flags

---

### 2. Correlation Analysis (`02_correlation_analysis.ipynb`)

**Goal:** Build a correlation-driven map of all sensor dependencies and compute the Correlation Health Index (CHI).

Methods used:
- Pearson, Spearman, and Kendall correlation matrices (45×45)
- Correlation Health Index: `CHI(s) = (1/N) Σ corr(s, j)` for all peer sensors j ≠ s
- Rolling 24-hour window correlations for all 990 undirected pairs
- Time-lagged correlations (±60 min) for selected pairs
- Mutual information via `mutual_info_regression` (scikit-learn)
- Granger causality tests on all 1,980 ordered pairs (maxlag=3, batched into 15 groups)
- Granger causality directed graph stored in GEXF format; eigenvector centrality computed
- Hierarchical clustering with Kendall's τ (average linkage)
- PCA-based dimensionality reduction (PC1 captures 99.97% of variance)

**Inputs:** `merged_for_correlation.csv`

**Outputs:**
- Pearson correlation heatmap
- CHI per subsystem with classification (Good / Moderate / Bad)
- Granger causality network in `/data/network/`
- PCA components and sensor grouping hierarchy

---

### 3. Statistical Diagnostics (`03_statistical_diagnostics.ipynb`)

**Goal:** Detect structural anomalies using classical statistical methods and compute Tier 1 outputs.

Techniques implemented:
- **Heartbeat check:** classifies each sensor as Dead / Frozen / Stuck / Alive based on timestamp gaps, first differences (|δvₜ| = |Vₜ − Vₜ₋₁|), and rolling variance
- **Sensor Reliability Index:** `SRI = Var(V) / (|μ(V)| + ε)` — normalised variance relative to operating point
- **Tukey fence detection:** upper fence = Q3 + 1.5 × IQR = 118.1 — flags pathologically unstable sensors
- **Dynamic z-score outlier detection:** per-sensor rolling window thresholds
- **Drift detection:** percentage of test-period readings outside training 3σ envelope

**Outputs:**
- Diagnostics tables → `/data/diagnostics/`
- Time-series plots with highlighted anomalies
- Heartbeat classification results for all 45 sensors
- SRI rankings and flagged sensor list

---

### 4. Outlier Verification (`04_outlier_verification.ipynb`)

**Goal:** Contextually validate detected anomalies using the ONV (Outlier Neighborhood Verification) layer.

Includes:
- Multi-sensor consistency checks using CHI-ranked peer groups
- ONV procedure: for each flagged observation, identify the 5 highest-CHI peers in the same subsystem, compute ONV_Agreement score, classify as "Sensor Fault" or "True Process Anomaly"
- Cross-verification of Isolation Forest flags with statistical outlier flags
- Final `Verified_Label` requires: Stat_Outlier AND IF_Flag == −1 AND ONV_Label == "Sensor Fault"

**Outputs:** `/data/diagnostics/outlier_verification/`

---

### 5. Machine Learning Integration (`05_ml_integration.ipynb`)

**Goal:** Deploy and train ML-based anomaly detectors, compute the Hybrid Health Score.

ML models used:
- **Denoising Autoencoder:** `Input(45) → GaussianNoise(0.02) → Dense(33,tanh,L2) → Dropout(0.1) → Dense(22,tanh,L2) → Dropout(0.1) → Dense(33,tanh,L2) → Dense(45,linear)` · Adam lr=1×10⁻⁴ · Early stopping patience=6 · Best val_loss = 0.04715
- **GRU Forecaster:** `Input(10,1) → GRU(64) → Dense(1,linear)` · Meta-model predicting next AE reconstruction error from previous 10 · Best val_loss = 0.01122
- **Isolation Forest:** contamination=0.02 · Trained on first chunk (2,349,692 rows) · Applied in 12 incremental chunks over full 28.2M training set
- **HHS fusion:** `HHS(t,i) = (1 − Norm(0.4 × AE_norm + 0.4 × GRU_norm + 0.2 × REG_norm)) × 100`

Key steps:
- StandardScaler normalisation (AE input) and min-max normalisation (HHS fusion)
- Temporal 80/20 train-validation split with 2% leakage-prevention gap
- Reconstruction error 3σ threshold = 0.990968

**Outputs:**
- Saved models → `/data/ml_models/checkpoints/` (`.keras` format, TensorFlow Serving compatible)
- Per-observation ML anomaly scores and HHS values → `/data/ml_models/`

---

### 6. Testing & Validation (`06_testing_models.ipynb`)

**Goal:** Evaluate all models on the held-out test set (17 Sep – 16 Nov 2025, 5,729,769 records).

Validation includes:
- Autoencoder reconstruction error distribution on test set (mean = 0.049197)
- GRU test MSE = 0.009687 / RMSE = 0.09842 / MAE = 0.007765
- Isolation Forest outlier ratio across test period (overall 8.67%)
- HHS temporal trend across all 45 sensors
- Cross-model corroboration of anomaly events (AE + GRU temporal coincidence)
- Drift detection results: high-drift sensors CT030.13 (2.16%), CT033.13 (2.04%), CT033.12 (2.04%)

**Outputs:**
- Model test metrics summary
- HHS temporal trend and per-sensor ranking (Figure 11 in thesis)
- Final performance comparison table

---

## Methodology Summary

| Stage | Method | Key Output |
|-------|--------|-----------|
| Preprocessing | Timestamp normalisation, z-score outlier removal, 1-min resampling, min-max normalisation | 601,510 clean observations (training) |
| Statistical diagnostics | Heartbeat check, SRI, Tukey fence | 45 Alive sensors; 5 high-SRI flagged |
| Correlation analysis | CHI, Granger causality, rolling correlations | Three-tier CHI subsystem hierarchy |
| ML anomaly detection | Denoising AE, GRU, Isolation Forest | 83 anomaly events; primary cluster 200× baseline |
| Contextual validation | ONV peer verification | Process events distinguished from sensor faults |
| Health scoring | HHS fusion (40-40-20) | Continuous 0–100 per-sensor reliability index |

---

## Dependencies

```bash
pip install -r requirements.txt
```

Python ≥ 3.9 recommended. Core dependencies: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `statsmodels`, `scipy`, `seaborn`, `matplotlib`, `networkx`.

All analyses performed in Python. Library versions and hardware details (Apple M4 GPU, Metal plugin, TensorFlow 2.x) are documented in Appendix A of the thesis.

---

## License

MIT License.

