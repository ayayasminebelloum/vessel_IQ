"""
Tier 1 — Statistical Diagnostics for the Tiered Graph Model (TGM).

Implements:
  - Heartbeat classification  (Dead / Frozen / Stuck / Alive)
  - Sensor Reliability Index  (SRI)
  - SRI classification        (good / moderate / bad)
  - Tukey fence detection     (pathologically unstable sensors)
  - Drift detection           (training-envelope exceedance on test set)
  - Tier-1 composite score    (CHI × 1/(1+SRI))

All thresholds match the thesis exactly:
  - Heartbeat gap threshold   : 30 minutes
  - Frozen threshold          : |δv| < 1e-5
  - Stuck threshold           : rolling variance (window=10) < 1e-5
  - SRI epsilon               : 1e-9
  - SRI percentile boundaries : P33 / P67
  - Tukey multiplier          : 1.5
  - Drift envelope            : 3σ of training distribution
"""

import os
import warnings
import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# Constants 

HEARTBEAT_GAP_MINUTES   = 30        # max allowable timestamp gap (minutes)
FROZEN_DELTA_THRESHOLD  = 1e-5      # |δv| threshold for "Frozen"
STUCK_VAR_THRESHOLD     = 1e-5      # rolling variance threshold for "Stuck"
STUCK_WINDOW            = 10        # rolling window for stuck detection
SRI_EPSILON             = 1e-9      # division-by-zero guard in SRI formula
DRIFT_SIGMA             = 3.0       # σ multiplier for drift envelope


# First step is HEARTBEAT CHECK

def heartbeat_check(df: pd.DataFrame, value_col: str = "Value") -> dict:
    """
    Classify a single sensor time-series into one of four operational states.

    States (checked in priority order):
        Dead   — maximum timestamp gap exceeds HEARTBEAT_GAP_MINUTES
        Frozen — all first differences |δv_t| = |V_t − V_{t-1}| < FROZEN_DELTA_THRESHOLD
        Stuck  — mean rolling variance (window=STUCK_WINDOW) < STUCK_VAR_THRESHOLD
        Alive  — passes all three tests
    """
  
    if df.empty or value_col not in df.columns:
        return {
            "status": "Dead",
            "max_gap_minutes": np.nan,
            "frozen_flag": True,
            "stuck_flag": True,
            "n_records": 0,
        }

    series = df[value_col].dropna()
    n = len(series)

    #  Check if dead  
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    time_diffs = df.index.to_series().diff().dt.total_seconds() / 60.0
    max_gap = time_diffs.max() if len(time_diffs) > 1 else 0.0

    if max_gap > HEARTBEAT_GAP_MINUTES:
        return {
            "status": "Dead",
            "max_gap_minutes": max_gap,
            "frozen_flag": False,
            "stuck_flag": False,
            "n_records": n,
        }

    # Check if frozen 
    # δv_t = V_t − V_{t-1}
    delta_v = series.diff().abs()
    frozen_flag = bool((delta_v.dropna() < FROZEN_DELTA_THRESHOLD).all())

    if frozen_flag:
        return {
            "status": "Frozen",
            "max_gap_minutes": max_gap,
            "frozen_flag": True,
            "stuck_flag": False,
            "n_records": n,
        }

    # Check if stuck 
    rolling_var = series.rolling(window=STUCK_WINDOW, min_periods=1).var()
    stuck_flag = bool(rolling_var.mean() < STUCK_VAR_THRESHOLD)

    if stuck_flag:
        return {
            "status": "Stuck",
            "max_gap_minutes": max_gap,
            "frozen_flag": False,
            "stuck_flag": True,
            "n_records": n,
        }

    return {
        "status": "Alive",
        "max_gap_minutes": max_gap,
        "frozen_flag": False,
        "stuck_flag": False,
        "n_records": n,
    }


def heartbeat_check_all(cleaned_dir: str, value_col: str = "Value") -> pd.DataFrame:
    # Run heartbeat_check on every *_cleaned.csv in cleaned_dir.
    rows = []
    files = [f for f in os.listdir(cleaned_dir) if f.endswith("_cleaned.csv")]

    for fname in sorted(files):
        sensor_name = fname.replace("_cleaned.csv", "")
        path = os.path.join(cleaned_dir, fname)
        try:
            df = pd.read_csv(path, parse_dates=["Timestamp"], index_col="Timestamp")
            result = heartbeat_check(df, value_col=value_col)
            result["sensor"] = sensor_name
            rows.append(result)
        except Exception as e:
            print(f"[heartbeat_check_all] Skipping {fname}: {e}")
            rows.append({
                "sensor": sensor_name,
                "status": "Error",
                "max_gap_minutes": np.nan,
                "frozen_flag": np.nan,
                "stuck_flag": np.nan,
                "n_records": 0,
            })

    df_out = pd.DataFrame(rows, columns=[
        "sensor", "status", "max_gap_minutes",
        "frozen_flag", "stuck_flag", "n_records",
    ])
    print(f"[heartbeat_check_all] Processed {len(df_out)} sensors.")
    print(df_out["status"].value_counts().to_string())
    return df_out


# 2. Sensor Reliability Index(SRI)

def compute_sri(series: pd.Series) -> float:
    """
    Compute the Sensor Reliability Index for a single value series.

        SRI = Var(V) / (|μ(V)| + ε)

    where ε = SRI_EPSILON prevents division by zero when the mean is near zero.
    A higher SRI indicates greater signal instability relative to operating point.
    """
    clean = series.dropna()
    if len(clean) < 2:
        return np.nan

    variance = float(clean.var(ddof=1))
    mean_abs = float(abs(clean.mean()))
    return variance / (mean_abs + SRI_EPSILON)


def compute_sri_all(cleaned_dir: str, value_col: str = "Value") -> pd.Series:
    # Compute SRI for every sensor in cleaned_dir.
    results = {}
    files = [f for f in os.listdir(cleaned_dir) if f.endswith("_cleaned.csv")]

    for fname in sorted(files):
        sensor_name = fname.replace("_cleaned.csv", "")
        path = os.path.join(cleaned_dir, fname)
        try:
            df = pd.read_csv(path, usecols=["Timestamp", value_col],
                             parse_dates=["Timestamp"])
            results[sensor_name] = compute_sri(df[value_col])
        except Exception as e:
            print(f"[compute_sri_all] Skipping {fname}: {e}")
            results[sensor_name] = np.nan

    sri_series = pd.Series(results, name="SRI").sort_values(ascending=False)
    print(f"[compute_sri_all] Computed SRI for {sri_series.notna().sum()} sensors.")
    return sri_series


# SRI Classification

def classify_sri(sri_series: pd.Series) -> pd.DataFrame:
    """
    Classify SRI values into three tiers using global distribution percentiles.

        below P33 → "good"
        P33–P67   → "moderate"
        above P67 → "bad"
    """
    clean = sri_series.dropna()
    p33 = float(clean.quantile(0.33))
    p67 = float(clean.quantile(0.67))

    def _classify(v):
        if pd.isna(v):
            return "unknown"
        if v < p33:
            return "good"
        if v <= p67:
            return "moderate"
        return "bad"

    df = pd.DataFrame({
        "sensor": sri_series.index,
        "SRI": sri_series.values,
        "classification": [_classify(v) for v in sri_series.values],
    })
    df["p33"] = p33
    df["p67"] = p67

    print(f"[classify_sri] P33={p33:.4f}  P67={p67:.4f}")
    print(df["classification"].value_counts().to_string())
    return df.reset_index(drop=True)


# TUKEY Fence Detection

def tukey_fence(sri_series: pd.Series, k: float = 1.5) -> dict:
    """
    Identify sensors with pathologically high SRI using the Tukey upper fence.

        upper_fence = Q3 + k × IQR

    With the dataset this yields upper_fence = 118.1 (k=1.5).
    """
    clean = sri_series.dropna()
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    upper_fence = q3 + k * iqr

    flagged = sri_series[sri_series > upper_fence].sort_values(ascending=False)

    print(f"[tukey_fence] Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}  "
          f"Upper fence={upper_fence:.2f}  Flagged={len(flagged)}")
    for sensor, val in flagged.items():
        print(f"  ✗ {sensor}: SRI = {val:.2f}")

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "upper_fence": upper_fence,
        "flagged": flagged,
        "n_flagged": len(flagged),
    }

# Drift Detection
def drift_detection(
    train_series: pd.Series,
    test_series: pd.Series,
    sigma: float = DRIFT_SIGMA,
) -> dict:
    """
    Detect sensor drift by measuring how many test-period readings fall
    outside the training distribution's σ-envelope.

    Method:
        1. Compute μ and σ from training series.
        2. Define envelope: [μ − sigma×σ,  μ + sigma×σ].
        3. Count test observations outside the envelope.
        4. Drift rate = outside_count / len(test_series).
    """
    train_clean = train_series.dropna()
    test_clean  = test_series.dropna()

    if len(train_clean) < 2 or len(test_clean) == 0:
        return {
            "train_mean": np.nan, "train_std": np.nan,
            "lower_bound": np.nan, "upper_bound": np.nan,
            "n_outside": 0, "n_test": len(test_clean),
            "drift_rate": np.nan, "drift_pct": np.nan,
            "drifting": False,
        }

    mu    = float(train_clean.mean())
    sigma_ = float(train_clean.std(ddof=1))
    lower = mu - sigma * sigma_
    upper = mu + sigma * sigma_

    outside = ((test_clean < lower) | (test_clean > upper)).sum()
    drift_rate = float(outside) / len(test_clean)

    return {
        "train_mean":  mu,
        "train_std":   sigma_,
        "lower_bound": lower,
        "upper_bound": upper,
        "n_outside":   int(outside),
        "n_test":      len(test_clean),
        "drift_rate":  drift_rate,
        "drift_pct":   drift_rate * 100,
        "drifting":    drift_rate > 0,
    }


def drift_detection_all(
    cleaned_dir: str,
    test_start: str = "2025-09-17",
    value_col: str = "Value",
    sigma: float = DRIFT_SIGMA,
) -> pd.DataFrame:
    # Run drift_detection for every sensor, splitting on test_start date.

    rows = []
    files = [f for f in os.listdir(cleaned_dir) if f.endswith("_cleaned.csv")]

    for fname in sorted(files):
        sensor_name = fname.replace("_cleaned.csv", "")
        path = os.path.join(cleaned_dir, fname)
        try:
            df = pd.read_csv(path, parse_dates=["Timestamp"],
                             index_col="Timestamp")
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            cutoff = pd.Timestamp(test_start, tz="UTC")
            train = df.loc[df.index < cutoff, value_col]
            test  = df.loc[df.index >= cutoff, value_col]
            result = drift_detection(train, test, sigma=sigma)
            result["sensor"] = sensor_name
            rows.append(result)
        except Exception as e:
            print(f"[drift_detection_all] Skipping {fname}: {e}")

    df_out = pd.DataFrame(rows).sort_values("drift_pct", ascending=False)
    df_out = df_out[["sensor", "drift_pct", "n_outside", "n_test",
                     "train_mean", "train_std", "lower_bound", "upper_bound",
                     "drifting"]]
    print(f"\n[drift_detection_all] Top drifting sensors:")
    print(df_out.head(10).to_string(index=False))
    return df_out.reset_index(drop=True)


# Calculate Composite Score

def tier1_composite(chi_series: pd.Series, sri_series: pd.Series) -> pd.Series:
    """
    Compute the Tier-1 diagnostic composite used for visualisation.

        score(s) = CHI(s) × 1 / (1 + SRI(s))

    Rewards high network correlation and low signal instability simultaneously.
    This score is superseded by the full Hybrid Health Score (HHS) in Tier 3.
    """
    aligned_chi, aligned_sri = chi_series.align(sri_series, join="inner")
    composite = aligned_chi * (1.0 / (1.0 + aligned_sri))
    return composite.sort_values(ascending=False).rename("tier1_composite")


# Running first tier function

def run_tier1(
    cleaned_dir: str,
    output_dir: str = "data/diagnostics",
    test_start: str = "2025-09-17",
    value_col: str = "Value",
) -> dict:
  
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("TIER 1 — STATISTICAL DIAGNOSTICS")
    print("=" * 60)

    # Heartbeat Check
    print("\n[1/4] Running heartbeat classification...")
    hb_df = heartbeat_check_all(cleaned_dir, value_col=value_col)
    hb_df.to_csv(os.path.join(output_dir, "heartbeat_results.csv"), index=False)

    # Calculate SRI 
    print("\n[2/4] Computing Sensor Reliability Index (SRI)...")
    sri = compute_sri_all(cleaned_dir, value_col=value_col)
    sri_classified = classify_sri(sri)
    sri_classified.to_csv(os.path.join(output_dir, "sri_classified.csv"), index=False)

    # Calculate Tukey fence 
    print("\n[3/4] Running Tukey fence detection...")
    tukey = tukey_fence(sri)
    flagged_df = tukey["flagged"].reset_index()
    flagged_df.columns = ["sensor", "SRI"]
    flagged_df.to_csv(os.path.join(output_dir, "sri_flagged.csv"), index=False)

    # Detect Drift  
    print(f"\n[4/4] Running drift detection (test start: {test_start})...")
    drift_df = drift_detection_all(
        cleaned_dir, test_start=test_start, value_col=value_col
    )
    drift_df.to_csv(os.path.join(output_dir, "drift_results.csv"), index=False)

    print("\n" + "=" * 60)
    print(f"Tier-1 complete. Results saved to: {output_dir}")
    print("=" * 60)

    return {
        "heartbeat_df":   hb_df,
        "sri_series":     sri,
        "sri_classified": sri_classified,
        "tukey_result":   tukey,
        "drift_df":       drift_df,
    }


# Main execution loop

if __name__ == "__main__":
    CLEANED_DIR = "data/cleaned_csv/cleaned"
    OUTPUT_DIR  = "data/diagnostics"
    TEST_START  = "2025-09-17"

    results = run_tier1(
        cleaned_dir=CLEANED_DIR,
        output_dir=OUTPUT_DIR,
        test_start=TEST_START,
    )
