"""
Evaluation, comparison, and final reporting for the Tiered Graph Model (TGM).

Implements:
  - Autoencoder evaluation    (reconstruction error stats, anomaly count)
  - GRU evaluation            (MSE, RMSE, MAE on test set)
  - Isolation Forest eval     (outlier ratio, daily trend)
  - Model comparison table    (matches Table 10 in the thesis)
  - HHS temporal analysis     (trend, sensor ranking, alert events)
  - Drift summary table       (matches Table 11 in the thesis)

Expected thesis values (for sanity-checking your run):
  AE  best val_loss            : 0.04715  (epoch 1)
  AE  test mean recon error    : 0.049197
  AE  3σ threshold             : 0.990968
  AE  test anomaly events      : 83
  GRU test MSE / RMSE / MAE   : 0.009687 / 0.09842 / 0.007765
  IF  overall outlier ratio    : 8.67%
  HHS stable baseline          : ~95/100
  HHS dip below alert (70)     : coincides with the anomaly cluster
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# Constants

HHS_ALERT_THRESHOLD = 70.0   # HHS below this triggers a maintenance advisory
ANOMALY_SIGMA       = 3.0    # σ multiplier for AE reconstruction error threshold


# Autoencoder Evaluation

def evaluate_autoencoder(
    test_errors: np.ndarray,
    threshold: float,
) -> dict:

    # Evaluate autoencoder performance on the test set.
    anomaly_mask    = test_errors > threshold
    anomaly_indices = np.where(anomaly_mask)[0]

    result = {
        "n_test":          len(test_errors),
        "mean_error":      float(test_errors.mean()),
        "std_error":       float(test_errors.std()),
        "max_error":       float(test_errors.max()),
        "threshold":       float(threshold),
        "n_anomalies":     int(anomaly_mask.sum()),
        "anomaly_rate":    float(anomaly_mask.mean()),
        "anomaly_indices": anomaly_indices,
    }

    print("[evaluate_autoencoder]")
    print(f"  Test samples      : {result['n_test']:,}")
    print(f"  Mean recon error  : {result['mean_error']:.6f}")
    print(f"  Std  recon error  : {result['std_error']:.6f}")
    print(f"  Max  recon error  : {result['max_error']:.6f}")
    print(f"  Threshold (3σ)    : {result['threshold']:.6f}")
    print(f"  Anomaly events    : {result['n_anomalies']}  "
          f"({result['anomaly_rate']:.4%})")

    if result["mean_error"] > 0:
        ratio = result["max_error"] / result["mean_error"]
        print(f"  Max / Mean ratio  : {ratio:.1f}×  "
              f"(thesis: ~200×)")

    return result


# GRU Evaluation

def evaluate_gru(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    # Evaluate GRU forecaster on test-period reconstruction errors.
  
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(y_true - y_pred)))

    result = {"mse": mse, "rmse": rmse, "mae": mae}

    print("[evaluate_gru]")
    print(f"  MSE  : {mse:.6f}  (thesis: 0.009687)")
    print(f"  RMSE : {rmse:.6f}  (thesis: 0.09842)")
    print(f"  MAE  : {mae:.6f}  (thesis: 0.007765)")

    return result


# Isolation Forest Evaluation

def evaluate_isolation_forest(
    if_flags: pd.Series,
) -> dict:

    # Evaluate Isolation Forest results on the test period.
    n_total    = len(if_flags)
    n_anom     = int((if_flags == -1).sum())
    ratio      = n_anom / n_total if n_total > 0 else 0.0

    # Daily outlier rate
    daily = (if_flags == -1).resample("1D").mean()

    print("[evaluate_isolation_forest]")
    print(f"  Total records   : {n_total:,}")
    print(f"  Anomalies       : {n_anom:,}")
    print(f"  Outlier ratio   : {ratio:.4%}  (thesis: 8.67%)")
    print(f"  Daily max ratio : {daily.max():.4%}")
    print(f"  Daily min ratio : {daily.min():.4%}")

    return {
        "n_total":       n_total,
        "n_anomalies":   n_anom,
        "outlier_ratio": ratio,
        "daily_trend":   daily,
    }


# Model Comparison Table

def compare_models(
    ae_result: dict,
    gru_result: dict,
    if_result: dict,
    hhs_result: dict,
    save_path: str = None,
) -> pd.DataFrame:
    # Produce the model performance comparison table.
    rows = [
        {
            "Model":          "Denoising Autoencoder",
            "Primary Metric": f"Val MSE = {ae_result.get('best_val_loss', 'N/A')}",
            "Test Metric":    f"Mean recon error = {ae_result['mean_error']:.6f}",
            "Threshold":      f"3σ = {ae_result['threshold']:.6f}",
            "Detections":     f"{ae_result['n_anomalies']} anomaly events",
            "Failure Mode":   "Shape anomalies (multivariate)",
        },
        {
            "Model":          "GRU Forecaster",
            "Primary Metric": f"Val MSE = {gru_result.get('val_mse', 'N/A')}",
            "Test Metric":    f"RMSE = {gru_result['rmse']:.5f}",
            "Threshold":      "Residual spike detection",
            "Detections":     "Independent confirmation of AE events",
            "Failure Mode":   "Temporal drift",
        },
        {
            "Model":          "Isolation Forest",
            "Primary Metric": f"Contamination = {0.02:.2f}",
            "Test Metric":    f"Outlier ratio = {if_result['outlier_ratio']:.4%}",
            "Threshold":      "contamination=0.02",
            "Detections":     f"{if_result['n_anomalies']:,} flagged records",
            "Failure Mode":   "Global multivariate deviation",
        },
        {
            "Model":          "Hybrid Health Score (TGM)",
            "Primary Metric": f"Coverage: {hhs_result.get('n_observations', 'N/A'):,} obs",
            "Test Metric":    f"Baseline ~{hhs_result.get('stable_mean', 95):.0f}/100",
            "Threshold":      "HHS < 70 = alert",
            "Detections":     f"{hhs_result.get('n_alerts', 0)} alert events",
            "Failure Mode":   "Shape + temporal + structural",
        },
    ]

    df = pd.DataFrame(rows)

    print("\n[compare_models] Model Performance Comparison")
    print(df.to_string(index=False))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"  Saved to: {save_path}")

    return df


# HHS Temporal Analysis

def analyse_hhs(
    hhs_series: pd.Series,
    alert_threshold: float = HHS_ALERT_THRESHOLD,
    save_path: str = None,
) -> dict:
    # Analyse Hybrid Health Score temporal behaviour and per-sensor ranking.
    n_obs      = len(hhs_series)
    alert_mask = hhs_series < alert_threshold
    n_alerts   = int(alert_mask.sum())
    stable     = hhs_series[~alert_mask]
    daily_mean = hhs_series.resample("1D").mean()

    result = {
        "n_observations":   n_obs,
        "mean_hhs":         float(hhs_series.mean()),
        "min_hhs":          float(hhs_series.min()),
        "max_hhs":          float(hhs_series.max()),
        "n_alerts":         n_alerts,
        "stable_mean":      float(stable.mean()) if len(stable) > 0 else np.nan,
        "alert_timestamps": hhs_series.index[alert_mask].tolist(),
        "daily_mean":       daily_mean,
    }

    print("[analyse_hhs]")
    print(f"  Observations      : {n_obs:,}")
    print(f"  Mean HHS          : {result['mean_hhs']:.2f}")
    print(f"  Min HHS           : {result['min_hhs']:.2f}")
    print(f"  Max HHS           : {result['max_hhs']:.2f}")
    print(f"  Stable baseline   : {result['stable_mean']:.2f}  (thesis: ~95)")
    print(f"  Alert events      : {n_alerts}  (HHS < {alert_threshold})")

    if save_path and PLOT_AVAILABLE:
        _plot_hhs_temporal(hhs_series, alert_threshold, save_path)

    return result


def sensor_hhs_ranking(
    hhs_per_sensor: pd.DataFrame,
    save_path: str = None,
) -> pd.DataFrame:
    # Compute mean HHS per sensor and return ranked table.
    ranking = pd.DataFrame({
        "sensor":   hhs_per_sensor.columns,
        "mean_hhs": hhs_per_sensor.mean().values,
        "min_hhs":  hhs_per_sensor.min().values,
    }).sort_values("mean_hhs")

    ranking["classification"] = pd.cut(
        ranking["mean_hhs"],
        bins=[0, 60, 80, 100],
        labels=["Critical", "Degraded", "Healthy"],
        right=True,
    )

    print("\n[sensor_hhs_ranking] Top 10 lowest-scoring sensors:")
    print(ranking.head(10).to_string(index=False))

    if save_path:
        ranking.to_csv(save_path, index=False)

    return ranking.reset_index(drop=True)


# Drift Summary Table

def drift_summary(
    drift_df: pd.DataFrame,
    top_n: int = 10,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Produce the drift detection results table (Table 11 in thesis).

    Expected top results:
        CT030.13  — 2.16%
        CT033.13  — 2.04%
        CT033.12  — 2.04%
    """
    cols = ["sensor", "drift_pct", "n_outside", "n_test",
            "train_mean", "train_std"]
    available = [c for c in cols if c in drift_df.columns]
    summary = drift_df[available].head(top_n).copy()

    if "drift_pct" in summary.columns:
        summary["drift_pct"] = summary["drift_pct"].round(4)

    print(f"\n[drift_summary] Top {top_n} drifting sensors:")
    print(summary.to_string(index=False))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        summary.to_csv(save_path, index=False)

    return summary


# Plotting Helper Function

def _plot_hhs_temporal(
    hhs_series: pd.Series,
    alert_threshold: float,
    save_path: str,
) -> None:
    # Plot HHS temporal trend with alert threshold line.
    if not PLOT_AVAILABLE:
        print("[_plot_hhs_temporal] matplotlib not available.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hhs_series.index, hhs_series.values,
            color="#2E75B6", linewidth=0.8, alpha=0.9, label="HHS")
    ax.axhline(alert_threshold, color="#C00000", linewidth=1.5,
               linestyle="--", label=f"Alert threshold ({alert_threshold})")
    ax.fill_between(hhs_series.index, hhs_series.values, alert_threshold,
                    where=hhs_series.values < alert_threshold,
                    color="#C00000", alpha=0.15, label="Alert zone")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hybrid Health Score (0–100)")
    ax.set_title("Hybrid Health Score — Temporal Trend (Test Period)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[_plot_hhs_temporal] Saved to: {save_path}")


def plot_reconstruction_errors(
    train_errors: np.ndarray,
    test_errors: np.ndarray,
    threshold: float,
    save_path: str = None,
) -> None:
    # Plot AE reconstruction error distribution and test-period time series.
    if not PLOT_AVAILABLE:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: distribution
    axes[0].hist(train_errors, bins=100, color="#2E75B6", alpha=0.7,
                 label="Train errors", density=True)
    axes[0].axvline(threshold, color="#C00000", linewidth=2,
                    linestyle="--", label=f"3σ threshold ({threshold:.4f})")
    axes[0].set_xlabel("Reconstruction MSE")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Training Error Distribution")
    axes[0].legend()

    # Right: test time series
    axes[1].plot(test_errors, color="#2E75B6", linewidth=0.6, alpha=0.8)
    axes[1].axhline(threshold, color="#C00000", linewidth=1.5,
                    linestyle="--", label=f"Threshold ({threshold:.4f})")
    anomalies = np.where(test_errors > threshold)[0]
    axes[1].scatter(anomalies, test_errors[anomalies],
                    color="#C00000", s=15, zorder=5, label="Anomalies")
    axes[1].set_xlabel("Test observation index")
    axes[1].set_ylabel("Reconstruction MSE")
    axes[1].set_title("Test Reconstruction Error")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_reconstruction_errors] Saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_gru_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None,
) -> None:
    # Plot GRU actual vs predicted reconstruction errors.
    if not PLOT_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_true, label="Actual", color="#2E75B6", linewidth=0.7, alpha=0.9)
    ax.plot(y_pred, label="Predicted", color="#ED7D31",
            linewidth=0.7, alpha=0.9, linestyle="--")
    ax.set_xlabel("Test step")
    ax.set_ylabel("Reconstruction error")
    ax.set_title("GRU Forecaster — Actual vs Predicted")
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_gru_predictions] Saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_if_daily_ratio(
    daily_trend: pd.Series,
    save_path: str = None,
) -> None:
    # Plot daily Isolation Forest outlier ratio over the test period.
    if not PLOT_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(daily_trend.index, daily_trend.values * 100,
           color="#2E75B6", alpha=0.8, width=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Outlier Ratio (%)")
    ax.set_title("Daily Outlier Ratio Over Time — Isolation Forest")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_if_daily_ratio] Saved to: {save_path}")
    else:
        plt.show()
    plt.close()


# Run Evaluation Function
def run_evaluation(
    ae_train_errors: np.ndarray,
    ae_test_errors: np.ndarray,
    ae_threshold: float,
    gru_y_true: np.ndarray,
    gru_y_pred: np.ndarray,
    if_flags: pd.Series,
    hhs_series: pd.Series,
    drift_df: pd.DataFrame,
    output_dir: str = "results",
    plot_dir: str = "images",
) -> dict:
    # Run the full evaluation pipeline and save all outputs.
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir,   exist_ok=True)

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # AE
    print("\n── Autoencoder ──")
    ae_result = evaluate_autoencoder(ae_test_errors, ae_threshold)
    plot_reconstruction_errors(
        ae_train_errors, ae_test_errors, ae_threshold,
        save_path=os.path.join(plot_dir, "ae_reconstruction_errors.png"),
    )

    # GRU
    print("\n── GRU Forecaster ──")
    gru_result = evaluate_gru(gru_y_true, gru_y_pred)
    plot_gru_predictions(
        gru_y_true, gru_y_pred,
        save_path=os.path.join(plot_dir, "gru_predictions.png"),
    )

    # IF
    print("\n── Isolation Forest ──")
    if_result = evaluate_isolation_forest(if_flags)
    plot_if_daily_ratio(
        if_result["daily_trend"],
        save_path=os.path.join(plot_dir, "if_daily_ratio.png"),
    )

    # HHS
    print("\n── Hybrid Health Score ──")
    hhs_result = analyse_hhs(
        hhs_series,
        save_path=os.path.join(plot_dir, "hhs_temporal.png"),
    )

    # Comparison table 
    print("\n── Model Comparison Table ──")
    comparison = compare_models(
        ae_result, gru_result, if_result, hhs_result,
        save_path=os.path.join(output_dir, "model_comparison.csv"),
    )

    # Drift summary 
    print("\n── Drift Detection Results ──")
    drift_table = drift_summary(
        drift_df,
        save_path=os.path.join(output_dir, "drift_summary.csv"),
    )

    print("\n" + "=" * 60)
    print(f"Evaluation complete. Results saved to: {output_dir}")
    print(f"Plots saved to: {plot_dir}")
    print("=" * 60)

    return {
        "ae_result":    ae_result,
        "gru_result":   gru_result,
        "if_result":    if_result,
        "hhs_result":   hhs_result,
        "comparison":   comparison,
        "drift_table":  drift_table,
    }


# Main Execution Loop

if __name__ == "__main__":
    print("model_testing.py — import this module from your notebooks.")
    print("Call run_evaluation(...) with outputs from ml_integration.py "
          "and statistical_tools.py.")