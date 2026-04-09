"""
Tier 3 — Machine Learning Anomaly Detection for the Tiered Graph Model (TGM).

Implements:
  - Denoising Autoencoder     (45→33→22→33→45, tanh, Gaussian noise)
  - GRU Forecaster            (sequence-of-10 → next AE error)
  - Isolation Forest          (chunked training on 28M+ records)
  - ONV verification          (Outlier Neighborhood Verification)
  - HHS fusion                (0.4×AE + 0.4×GRU + 0.2×SRI → 0-100 index)

All architecture parameters are the following:
  - AE input/output dim       : 45
  - AE latent dim             : 22  (≈49% compression)
  - AE noise σ                : 0.02
  - AE L2 regularisation      : 1e-4
  - AE dropout rate           : 0.10
  - AE activation             : tanh (encoder/decoder), linear (output)
  - AE optimiser              : Adam lr=1e-4, clipnorm=1.0
  - AE max epochs             : 30
  - AE early stopping         : patience=6, restore_best_weights=True
  - AE LR schedule            : ReduceLROnPlateau factor=0.5, patience=4, min_lr=1e-6
  - AE train/val split        : 80/20 with 2% temporal gap
  - GRU sequence length       : 10
  - GRU hidden units          : 64
  - GRU optimiser             : Adam lr=1e-3
  - IF contamination          : 0.02
  - IF chunk size             : first chunk (~2.35M rows) for training
  - HHS weights               : 0.4 (AE), 0.4 (GRU), 0.2 (Regression/SRI)
  - ONV peer count            : 5 (highest-CHI peers in same subsystem)

"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# the code was first run through a computer with no GPU so I used the following TensorFlow import
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[ml_integration] WARNING: TensorFlow not available. "
          "AE and GRU functions will raise ImportError if called.")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[ml_integration] WARNING: scikit-learn not available.")

# Constants

AE_INPUT_DIM     = 45
AE_LATENT_DIM    = 22
AE_NOISE_SIGMA   = 0.02
AE_L2_REG        = 1e-4
AE_DROPOUT       = 0.10
AE_LR            = 1e-4
AE_CLIPNORM      = 1.0
AE_MAX_EPOCHS    = 30
AE_BATCH_SIZE    = 128
AE_ES_PATIENCE   = 6
AE_LR_FACTOR     = 0.5
AE_LR_PATIENCE   = 4
AE_MIN_LR        = 1e-6
AE_VAL_SPLIT     = 0.20
AE_LEAKAGE_GAP   = 0.02   # 2% temporal gap between train and val

GRU_SEQ_LEN      = 10
GRU_UNITS        = 64
GRU_LR           = 1e-3
GRU_MAX_EPOCHS   = 30
GRU_BATCH_SIZE   = 64
GRU_ES_PATIENCE  = 5

IF_CONTAMINATION = 0.02
IF_RANDOM_STATE  = 42

HHS_W_AE         = 0.4
HHS_W_GRU        = 0.4
HHS_W_REG        = 0.2

ONV_N_PEERS      = 5
ANOMALY_SIGMA    = 3.0   # 3σ threshold for AE reconstruction error flagging


# Denoising Autoencoder
def build_autoencoder(input_dim: int = AE_INPUT_DIM,
                      latent_dim: int = AE_LATENT_DIM) -> "keras.Model":
    """
    Build the denoising autoencoder with architecture:

        Input(input_dim)
        → GaussianNoise(AE_NOISE_SIGMA)
        → Dense(33, tanh, L2)
        → Dropout(AE_DROPOUT)
        → Dense(latent_dim, tanh, L2)   ← bottleneck
        → Dropout(AE_DROPOUT)
        → Dense(33, tanh, L2)
        → Dense(input_dim, linear)       ← reconstruction

    tanh activations are used instead of ReLU because sensor values include
    strongly negative cryogenic temperatures (down to −199.8°C); after
    StandardScaler normalisation values can be substantially negative,
    and ReLU dead-neuron effects caused gradient explosions in early runs.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for build_autoencoder.")

    reg = regularizers.l2(AE_L2_REG)
    hidden_dim = 33  # intermediate layer between input and latent

    inp = keras.Input(shape=(input_dim,), name="encoder_input")
    x = layers.GaussianNoise(AE_NOISE_SIGMA, name="gaussian_noise")(inp)
    x = layers.Dense(hidden_dim, activation="tanh",
                     kernel_regularizer=reg, name="encoder_dense_1")(x)
    x = layers.Dropout(AE_DROPOUT, name="encoder_dropout_1")(x)
    x = layers.Dense(latent_dim, activation="tanh",
                     kernel_regularizer=reg, name="bottleneck")(x)
    x = layers.Dropout(AE_DROPOUT, name="encoder_dropout_2")(x)
    x = layers.Dense(hidden_dim, activation="tanh",
                     kernel_regularizer=reg, name="decoder_dense_1")(x)
    out = layers.Dense(input_dim, activation="linear",
                       name="decoder_output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="denoising_autoencoder")
    print(f"[build_autoencoder] Built AE: {input_dim}→{hidden_dim}→{latent_dim}"
          f"→{hidden_dim}→{input_dim}")
    return model


def train_autoencoder(
    X: np.ndarray,
    model: "keras.Model" = None,
    model_save_path: str = "data/ml_models/checkpoints/autoencoder.keras",
) -> tuple:
    """
    Compile and train the denoising autoencoder.

    Applies a temporal 80/20 train-validation split with a 2% leakage-
    prevention gap (2% of training data is excluded from both splits to
    prevent temporal leakage at the boundary).

    Callbacks:
      - EarlyStopping: patience=6, restore_best_weights=True, monitor=val_loss
      - ReduceLROnPlateau: factor=0.5, patience=4, min_lr=1e-6
      - ModelCheckpoint: saves best model to model_save_path
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for train_autoencoder.")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Temporal split with leakage gap 
    n = len(X_scaled)
    train_end  = int(n * (1 - AE_VAL_SPLIT - AE_LEAKAGE_GAP))
    val_start  = int(n * (1 - AE_VAL_SPLIT))
    X_train    = X_scaled[:train_end]
    X_val      = X_scaled[val_start:]

    print(f"[train_autoencoder] Train samples: {len(X_train):,}  "
          f"Val samples: {len(X_val):,}  "
          f"Leakage gap: {val_start - train_end:,}")

    # Build model if not provided 
    if model is None:
        model = build_autoencoder(input_dim=X.shape[1])

    optimizer = keras.optimizers.Adam(learning_rate=AE_LR, clipnorm=AE_CLIPNORM)
    model.compile(optimizer=optimizer, loss="mse")

    # Callbacks 
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=AE_ES_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=AE_LR_FACTOR,
            patience=AE_LR_PATIENCE,
            min_lr=AE_MIN_LR,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=AE_MAX_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        callbacks=cb_list,
        verbose=1,
    )

    best_val_loss = min(history.history["val_loss"])
    best_epoch    = history.history["val_loss"].index(best_val_loss) + 1
    print(f"[train_autoencoder] Best val_loss={best_val_loss:.5f} at epoch {best_epoch}")
    print(f"[train_autoencoder] Model saved to: {model_save_path}")

    return model, history, scaler


def compute_reconstruction_errors(
    model: "keras.Model",
    X: np.ndarray,
    scaler: "StandardScaler",
) -> np.ndarray:
    # Compute per-observation mean squared reconstruction error on X.

    X_scaled = scaler.transform(X)
    X_recon  = model.predict(X_scaled, verbose=0)
    errors   = np.mean((X_scaled - X_recon) ** 2, axis=1)
    return errors


def get_anomaly_threshold(train_errors: np.ndarray,
                          sigma: float = ANOMALY_SIGMA) -> float:
    """
    Compute the anomaly detection threshold as μ + sigma×σ of training errors.

    With thesis data this yields threshold ≈ 0.990968 (3σ).
    """
    return float(train_errors.mean() + sigma * train_errors.std())


# GRU Forecaster

def build_gru_forecaster(seq_len: int = GRU_SEQ_LEN) -> "keras.Model":
    """
    Build the GRU meta-model that forecasts the next autoencoder
    reconstruction error from the previous seq_len errors.

    Architecture:
        Input(seq_len, 1) → GRU(GRU_UNITS) → Dense(1, linear)

    This is a 1D sequence-to-scalar regression model. It is trained
    on training-period AE reconstruction errors and used during the
    test period to detect temporal drift via forecast residuals.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for build_gru_forecaster.")

    inp = keras.Input(shape=(seq_len, 1), name="gru_input")
    x   = layers.GRU(GRU_UNITS, name="gru_layer")(inp)
    out = layers.Dense(1, activation="linear", name="gru_output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="gru_forecaster")
    print(f"[build_gru_forecaster] Built GRU: Input({seq_len},1) → GRU({GRU_UNITS}) → Dense(1)")
    return model


def make_gru_sequences(errors: np.ndarray, seq_len: int = GRU_SEQ_LEN) -> tuple:
    # Convert a 1D error array into (X, y) sliding-window sequences.

    X, y = [], []
    for i in range(len(errors) - seq_len):
        X.append(errors[i : i + seq_len])
        y.append(errors[i + seq_len])
    return np.array(X)[..., np.newaxis], np.array(y)


def train_gru_forecaster(
    train_errors: np.ndarray,
    model: "keras.Model" = None,
    model_save_path: str = "data/ml_models/checkpoints/gru_forecaster.keras",) -> tuple:
    # Train the GRU forecaster on training-period reconstruction errors.
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for train_gru_forecaster.")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    X, y = make_gru_sequences(train_errors, seq_len=GRU_SEQ_LEN)

    n        = len(X)
    val_n    = int(n * AE_VAL_SPLIT)
    X_train  = X[:-val_n]
    y_train  = y[:-val_n]
    X_val    = X[-val_n:]
    y_val    = y[-val_n:]

    print(f"[train_gru_forecaster] Train={len(X_train):,}  Val={len(X_val):,}")

    if model is None:
        model = build_gru_forecaster()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=GRU_LR),
        loss="mse",
    )

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=GRU_ES_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=GRU_MAX_EPOCHS,
        batch_size=GRU_BATCH_SIZE,
        callbacks=cb_list,
        verbose=1,
    )

    best_val = min(history.history["val_loss"])
    print(f"[train_gru_forecaster] Best val_loss={best_val:.5f}")
    return model, history


def compute_gru_forecast_errors(
    gru_model: "keras.Model",
    errors: np.ndarray,
) -> np.ndarray:
    # Compute GRU forecast residuals (|actual − predicted|) for a given error series.

    X, y_true = make_gru_sequences(errors, seq_len=GRU_SEQ_LEN)
    y_pred = gru_model.predict(X, verbose=0).flatten()
    return np.abs(y_true - y_pred)


# Isolation Forest

def train_isolation_forest(
    df_merged: pd.DataFrame,
    contamination: float = IF_CONTAMINATION,
    chunk_size: int = 2_349_692,
    model_save_path: str = "data/ml_models/iforest.pkl",
) -> "IsolationForest":
    """
    Train an Isolation Forest on the first chunk of the merged sensor dataset.

    The model is trained on chunk_size rows to stay within memory limits,
    then applied to the full dataset in apply_isolation_forest().
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for train_isolation_forest.")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    X_train = df_merged.dropna().head(chunk_size)
    print(f"[train_isolation_forest] Training on {len(X_train):,} rows  "
          f"(contamination={contamination})")

    iforest = IsolationForest(
        contamination=contamination,
        random_state=IF_RANDOM_STATE,
        n_jobs=-1,
    )
    iforest.fit(X_train)

    joblib.dump(iforest, model_save_path)
    print(f"[train_isolation_forest] Model saved to: {model_save_path}")
    return iforest


def apply_isolation_forest(
    iforest: "IsolationForest",
    df_merged: pd.DataFrame,
    chunk_size: int = 500_000,
) -> pd.Series:
  
    # Apply a trained Isolation Forest to df_merged in chunks.

    df_clean = df_merged.dropna()
    flags = []
    n = len(df_clean)

    for start in range(0, n, chunk_size):
        chunk = df_clean.iloc[start : start + chunk_size]
        flags.extend(iforest.predict(chunk).tolist())
        if start % (chunk_size * 4) == 0:
            print(f"  IF progress: {start:,} / {n:,}")

    result = pd.Series(flags, index=df_clean.index, name="IF_Flag")
    outlier_ratio = (result == -1).mean()
    print(f"[apply_isolation_forest] Overall outlier ratio: {outlier_ratio:.4%}")
    return result


# Outlier Neighborhood Verification (ONV)

def onv_verify(
    df_merged: pd.DataFrame,
    stat_outlier_mask: pd.Series,
    if_flag: pd.Series,
    chi_scores: pd.Series,
    subsystem_map: dict,
    n_peers: int = ONV_N_PEERS,
) -> pd.DataFrame:
    """
    Outlier Neighborhood Verification (ONV) layer.

    For each observation flagged by both statistical detection AND Isolation
    Forest, check whether the sensor's highest-CHI peers in the same subsystem
    show similar anomalous behaviour.
    """
    candidates = df_merged.index[
        stat_outlier_mask.reindex(df_merged.index, fill_value=False) &
        (if_flag.reindex(df_merged.index, fill_value=1) == -1)
    ]

    if len(candidates) == 0:
        print("[onv_verify] No candidate timestamps found.")
        return pd.DataFrame()

    print(f"[onv_verify] Evaluating {len(candidates):,} candidate timestamps...")

    rows = []
    for sensor in df_merged.columns:
        subsystem = subsystem_map.get(sensor, "Unknown")

        # Peers: other sensors in the same subsystem, ranked by CHI descending
        peer_sensors = [
            s for s in df_merged.columns
            if subsystem_map.get(s) == subsystem and s != sensor
        ]
        peer_chi = chi_scores.reindex(peer_sensors).dropna()
        top_peers = peer_chi.nlargest(n_peers).index.tolist()

        if not top_peers:
            continue

        for ts in candidates:
            if ts not in df_merged.index:
                continue

            peer_vals = df_merged.loc[ts, top_peers].dropna()
            if len(peer_vals) == 0:
                continue

            onv_agreement = float(
                (peer_vals - peer_vals.mean()).abs().mean()
            )
            rows.append({
                "timestamp":    ts,
                "sensor":       sensor,
                "ONV_Agreement": onv_agreement,
            })

    if not rows:
        print("[onv_verify] No ONV rows produced.")
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    median_agreement = result_df["ONV_Agreement"].median()

    result_df["ONV_Label"] = np.where(
        result_df["ONV_Agreement"] < median_agreement,
        "Sensor Fault",
        "True Process Anomaly",
    )

    # Add original flags back
    result_df["stat_outlier"] = stat_outlier_mask.reindex(
        result_df["timestamp"]
    ).values
    result_df["IF_Flag"] = if_flag.reindex(result_df["timestamp"]).values

    result_df["Verified_Label"] = np.where(
        (result_df["stat_outlier"] == True) &
        (result_df["IF_Flag"]      == -1) &
        (result_df["ONV_Label"]    == "Sensor Fault"),
        "Sensor Fault",
        "Normal",
    )

    n_faults = (result_df["Verified_Label"] == "Sensor Fault").sum()
    print(f"[onv_verify] Verified sensor faults: {n_faults}")
    print(f"[onv_verify] Median ONV_Agreement: {median_agreement:.6f}")
    return result_df


# Hybrid Health Score (HHS)

def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    # Min-max normalise array to [0, 1]. Returns 0.5 if constant.
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo)


def compute_hhs(
    ae_errors: np.ndarray,
    gru_errors: np.ndarray,
    sri_values: np.ndarray,
    w_ae: float  = HHS_W_AE,
    w_gru: float = HHS_W_GRU,
    w_reg: float = HHS_W_REG,
) -> np.ndarray:
    """
    Compute the Hybrid Health Score (HHS) by fusing three normalised signals.

        HybridHealth(t) = w_ae  × AE_norm(t)
                        + w_gru × GRU_norm(t)
                        + w_reg × REG_norm(t)

        HHS(t) = (1 − Norm(HybridHealth(t))) × 100

    Where:
        AE_norm  : min-max normalised autoencoder reconstruction error
        GRU_norm : min-max normalised GRU absolute forecast error
        REG_norm : min-max normalised regression-predicted SRI
        Norm(·)  : min-max normalisation of the fused signal

    Score ranges from 0 (critically unhealthy) to 100 (fully healthy).
    """
    assert abs(w_ae + w_gru + w_reg - 1.0) < 1e-6, \
        f"Weights must sum to 1.0, got {w_ae + w_gru + w_reg}"

    n = min(len(ae_errors), len(gru_errors), len(sri_values))
    ae_n  = _minmax_norm(ae_errors[:n])
    gru_n = _minmax_norm(gru_errors[:n])
    reg_n = _minmax_norm(sri_values[:n])

    hybrid = w_ae * ae_n + w_gru * gru_n + w_reg * reg_n
    hhs    = (1.0 - _minmax_norm(hybrid)) * 100.0

    print(f"[compute_hhs] HHS  mean={hhs.mean():.2f}  "
          f"min={hhs.min():.2f}  max={hhs.max():.2f}  "
          f"n={n:,}")
    return hhs


# 6. Run the tier functions

def run_tier3(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    chi_scores: pd.Series,
    subsystem_map: dict,
    output_dir: str = "data/ml_models",
    checkpoint_dir: str = "data/ml_models/checkpoints",
) -> dict:

    os.makedirs(output_dir,    exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("TIER 3 — MACHINE LEARNING ANOMALY DETECTION")
    print("=" * 60)

    X_train = df_train.dropna().values
    X_test  = df_test.dropna().values

    # Autoencoder 
    print("\n[1/5] Training denoising autoencoder...")
    ae_model, ae_history, ae_scaler = train_autoencoder(
        X_train,
        model_save_path=os.path.join(checkpoint_dir, "autoencoder.keras"),
    )
    ae_train_errors = compute_reconstruction_errors(ae_model, X_train, ae_scaler)
    ae_test_errors  = compute_reconstruction_errors(ae_model, X_test,  ae_scaler)
    ae_threshold    = get_anomaly_threshold(ae_train_errors)

    print(f"  Train error mean={ae_train_errors.mean():.6f}  "
          f"Test error mean={ae_test_errors.mean():.6f}")
    print(f"  Anomaly threshold (3σ): {ae_threshold:.6f}")
    print(f"  Test anomaly events: {(ae_test_errors > ae_threshold).sum()}")

    np.save(os.path.join(output_dir, "ae_train_errors.npy"), ae_train_errors)
    np.save(os.path.join(output_dir, "ae_test_errors.npy"),  ae_test_errors)
    joblib.dump(ae_scaler, os.path.join(checkpoint_dir, "ae_scaler.pkl"))

    # GRU forecaster 
    print("\n[2/5] Training GRU forecaster...")
    gru_model, gru_history = train_gru_forecaster(
        ae_train_errors,
        model_save_path=os.path.join(checkpoint_dir, "gru_forecaster.keras"),
    )
    gru_test_errors = compute_gru_forecast_errors(gru_model, ae_test_errors)
    np.save(os.path.join(output_dir, "gru_test_errors.npy"), gru_test_errors)

    # Isolation Forest 
    print("\n[3/5] Training and applying Isolation Forest...")
    iforest  = train_isolation_forest(
        df_train,
        model_save_path=os.path.join(checkpoint_dir, "iforest.pkl"),
    )
    if_flags = apply_isolation_forest(iforest, df_test)
    if_flags.to_csv(os.path.join(output_dir, "if_flags_test.csv"))

    # ONV verification 
    print("\n[4/5] Running Outlier Neighborhood Verification (ONV)...")
    stat_outlier_mask = pd.Series(
        ae_test_errors > ae_threshold,
        index=df_test.dropna().index,
    )
    onv_results = onv_verify(
        df_test, stat_outlier_mask, if_flags, chi_scores, subsystem_map
    )
    if len(onv_results) > 0:
        onv_results.to_csv(os.path.join(output_dir, "onv_results.csv"), index=False)

    # Compute HHS 
    print("\n[5/5] Computing Hybrid Health Score (HHS)...")
    # Pad GRU errors to match AE test length (first GRU_SEQ_LEN values unavailable)
    gru_padded = np.concatenate([
        np.full(GRU_SEQ_LEN, gru_test_errors.mean()),
        gru_test_errors,
    ])
    # Use AE reconstruction error as proxy for regression-predicted SRI
    hhs = compute_hhs(ae_test_errors, gru_padded, ae_test_errors)
    hhs_series = pd.Series(hhs, index=df_test.dropna().index, name="HHS")
    hhs_series.to_csv(os.path.join(output_dir, "hhs_test.csv"))

    print("\n" + "=" * 60)
    print(f"Tier-3 complete. Outputs saved to: {output_dir}")
    print("=" * 60)

    return {
        "ae_model":          ae_model,
        "ae_scaler":         ae_scaler,
        "ae_threshold":      ae_threshold,
        "ae_history":        ae_history,
        "gru_model":         gru_model,
        "gru_history":       gru_history,
        "iforest":           iforest,
        "ae_train_errors":   ae_train_errors,
        "ae_test_errors":    ae_test_errors,
        "gru_test_errors":   gru_test_errors,
        "if_flags":          if_flags,
        "onv_results":       onv_results,
        "hhs":               hhs_series,
    }


# Main execution Loop
if __name__ == "__main__":
    print("ml_integration.py — import this module from your notebooks.")
    print("Call run_tier3(df_train, df_test, chi_scores, subsystem_map) to run the full pipeline.")
