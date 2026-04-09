import pandas as pd
import numpy as np
import os
from scipy import stats

# Configuration: Replace with your own paths
DATA_DIR = "data/raw"
BASE_OUTPUT_DIR = "data/cleaned_csv"
CLEANED_DIR = os.path.join(BASE_OUTPUT_DIR, "cleaned")
OUTLIERS_DIR = os.path.join(BASE_OUTPUT_DIR, "outliers")

os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(OUTLIERS_DIR, exist_ok=True)

RESAMPLE_RULE = "1T"             # resample interval (1 minute)
MAX_GAP_MINUTES = 30             # max gap interpolation limit


# Function to clean one sensor
def preprocess_sensor(file_path):
    sensor_name = os.path.basename(file_path)
    print(f"\nProcessing {sensor_name} ...")

    # Read and standardize 
    df = pd.read_csv(file_path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    #  Detect and parse timestamp
    time_col = next((c for c in df.columns if "time" in c.lower()), None)
    if time_col is None:
        print(f"Skipping {sensor_name} (no timestamp column)")
        return None

    df.rename(columns={time_col: "Timestamp"}, inplace=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    df = df.set_index("Timestamp")

    # Ensure numeric columns
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["ValueStatus"] = pd.to_numeric(df.get("ValueStatus", np.nan), errors="coerce")

    # Keep only ValueStatus > 0 
    before = len(df)
    df = df[df["ValueStatus"] > 0]
    dropped = before - len(df)
    print(f"Dropped {dropped} invalid rows (ValueStatus <= 0)")

    if df.empty:
        print(f"{sensor_name} has no valid data after filtering.")
        return None

    # Resample (mean per minute)
    df = df.resample(RESAMPLE_RULE).mean(numeric_only=True)

    # Interpolate Value + fill ValueStatus 
    if "Value" in df.columns:
        df["Value"] = df["Value"].interpolate(limit=int(MAX_GAP_MINUTES),
                                              limit_direction="both")
    if "ValueStatus" in df.columns:
        df["ValueStatus"] = df["ValueStatus"].ffill().bfill()

    # Dynamic outlier detection 
    if df["Value"].notna().sum() > 0:
        z_scores = np.abs(stats.zscore(df["Value"].fillna(method="ffill"), nan_policy="omit"))
        dynamic_z = np.nanmean(z_scores) + 3 * np.nanstd(z_scores)
        df["IsOutlier"] = z_scores > dynamic_z
    else:
        df["IsOutlier"] = False
        dynamic_z = 3.0

    # Save outliers separately
    outlier_df = df[df["IsOutlier"]]
    if not outlier_df.empty:
        outlier_path = os.path.join(OUTLIERS_DIR, sensor_name.replace(".csv", "_outliers.csv"))
        outlier_df.to_csv(outlier_path)
        print(f"Saved {len(outlier_df)} outliers to: {outlier_path}")

    #  Normalize Value and ValueStatus 
    if df["Value"].notna().sum() > 0:
        v_min, v_max = df["Value"].min(), df["Value"].max()
        df["Value_norm"] = (df["Value"] - v_min) / (v_max - v_min) if v_max != v_min else 0.5
    else:
        df["Value_norm"] = np.nan

    if df["ValueStatus"].notna().sum() > 0:
        vs_min, vs_max = df["ValueStatus"].min(), df["ValueStatus"].max()
        df["ValueStatus_norm"] = (df["ValueStatus"] - vs_min) / (vs_max - vs_min) if vs_max != vs_min else 1.0
    else:
        df["ValueStatus_norm"] = np.nan

    # drop ValueStatus_norm if constant
    if df["ValueStatus"].nunique() <= 1:
        df.drop(columns=["ValueStatus_norm"], inplace=True, errors="ignore")

    # Derived temporal features 
    df["RollingMean_10min"] = df["Value_norm"].rolling(window=10, min_periods=1).mean()
    df["RateOfChange"] = df["Value_norm"].diff()
    df["DailyMean"] = df["Value_norm"].rolling(window=1440, min_periods=1).mean()
    df["time_diff"] = df.index.to_series().diff().dt.total_seconds().fillna(0)

    #  Save cleaned file 
    cleaned_path = os.path.join(CLEANED_DIR, sensor_name.replace(".csv", "_cleaned.csv"))
    df.to_csv(cleaned_path)

    print(f"Saved cleaned file: {cleaned_path}")
    print(f"Non-null Value count: {df['Value'].notna().sum()}, unique Values: {df['Value'].nunique()}")
    print(f"Non-null ValueStatus count: {df['ValueStatus'].notna().sum()}, unique ValueStatus: {df['ValueStatus'].nunique()}")
    print(f"Outlier threshold (dynamic): {dynamic_z:.3f}")

    return df


# Loop over all files
all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

print(f"\nFound {len(all_files)} CSV files to process.")
for file in all_files:
    file_path = os.path.join(DATA_DIR, file)
    try:
        preprocess_sensor(file_path)
    except Exception as e:
        print(f"Error processing {file}: {e}")

print("\nBatch cleaning complete.")
print("Cleaned files saved to:", CLEANED_DIR)
print("Outliers saved to:", OUTLIERS_DIR)
