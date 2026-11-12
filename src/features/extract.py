"""
Feature extraction from windowed IMU signals.

Produces a flat feature vector per window suitable for classical ML.
Features follow established fall detection literature (Özdemir & Barshan, 2014;
Aziz et al., 2017) covering time-domain statistics and frequency-domain descriptors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis


# ---------------------------------------------------------------------------
# Single-window feature extraction
# ---------------------------------------------------------------------------

def _time_features(x: np.ndarray, prefix: str) -> dict:
    """Time-domain statistics for a 1-D signal x."""
    return {
        f"{prefix}_mean":        x.mean(),
        f"{prefix}_std":         x.std(),
        f"{prefix}_min":         x.min(),
        f"{prefix}_max":         x.max(),
        f"{prefix}_range":       x.max() - x.min(),
        f"{prefix}_iqr":         float(np.percentile(x, 75) - np.percentile(x, 25)),
        f"{prefix}_skewness":    float(skew(x)),
        f"{prefix}_kurtosis":    float(kurtosis(x)),
        f"{prefix}_rms":         float(np.sqrt(np.mean(x ** 2))),
        f"{prefix}_mad":         float(np.mean(np.abs(x - x.mean()))),
        f"{prefix}_energy":      float(np.sum(x ** 2)),
        f"{prefix}_zero_cross":  int(np.sum(np.diff(np.sign(x - x.mean())) != 0)),
    }


def _freq_features(x: np.ndarray, fs: int, prefix: str, n_bands: int = 5) -> dict:
    """Frequency-domain features for a 1-D signal x."""
    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=min(128, len(x)))
    total_power = psd.sum() + 1e-12

    dominant_freq = freqs[np.argmax(psd)]
    spectral_entropy = -np.sum((psd / total_power) * np.log(psd / total_power + 1e-12))
    spectral_centroid = np.sum(freqs * psd) / total_power

    # Band power (equal-width bins up to Nyquist)
    nyq = fs / 2
    band_edges = np.linspace(0, nyq, n_bands + 1)
    band_powers = {}
    for i in range(n_bands):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        band_powers[f"{prefix}_bp_{i}"] = float(psd[mask].sum())

    return {
        f"{prefix}_dom_freq":          float(dominant_freq),
        f"{prefix}_spectral_entropy":  float(spectral_entropy),
        f"{prefix}_spectral_centroid": float(spectral_centroid),
        f"{prefix}_total_power":       float(total_power),
        **band_powers,
    }


def extract_window_features(window: np.ndarray, fs: int) -> dict:
    """
    Extract all features from a single window.

    Parameters
    ----------
    window : (window_len, n_channels)
        Calibrated sensor data. Expected channel order:
        0–2 → ax, ay, az (g)
        3–5 → gx, gy, gz (°/s)   [optional]
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    dict mapping feature name → scalar float.
    """
    feats: dict = {}

    n_ch = window.shape[1] if window.ndim > 1 else 1
    channel_names = ["ax", "ay", "az", "gx", "gy", "gz"][:n_ch]

    # Per-channel features
    for i, ch in enumerate(channel_names):
        x = window[:, i]
        feats.update(_time_features(x, ch))
        feats.update(_freq_features(x, fs, ch))

    # SVM (resultant acceleration)
    acc = window[:, :min(3, n_ch)]
    svm = np.linalg.norm(acc, axis=1)
    feats.update(_time_features(svm, "svm"))
    feats.update(_freq_features(svm, fs, "svm"))

    # Jerk (first difference of SVM, in units/s)
    jerk = np.diff(svm) * fs
    feats.update(_time_features(jerk, "jerk"))

    # Net acceleration (remove gravity)
    net = svm - 1.0  # assumes g ≈ 1
    feats.update(_time_features(net, "net_acc"))

    # Correlation between axes (posture proxy)
    if n_ch >= 3:
        for pair, (i, j) in [("ax_ay", (0, 1)), ("ax_az", (0, 2)), ("ay_az", (1, 2))]:
            xi, xj = window[:, i], window[:, j]
            corr = float(np.corrcoef(xi, xj)[0, 1]) if xi.std() > 0 and xj.std() > 0 else 0.0
            feats[f"corr_{pair}"] = corr

    return feats


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_dataset_features(
    X: np.ndarray,
    y: np.ndarray,
    subjects: list[str],
    groups: list[str],
    fs: int,
) -> pd.DataFrame:
    """
    Extract features for every window in X.

    Parameters
    ----------
    X        : (N, window_len, n_channels)
    y        : (N,) binary fall labels
    subjects : (N,) subject IDs
    groups   : (N,) age group strings

    Returns
    -------
    DataFrame with one row per window; columns = features + meta columns.
    """
    rows = []
    for i in range(len(X)):
        feats = extract_window_features(X[i], fs)
        feats["label"]      = int(y[i])
        feats["subject_id"] = subjects[i]
        feats["age_group"]  = groups[i]
        rows.append(feats)

    df = pd.DataFrame(rows)
    # Move meta columns to front
    meta = ["label", "subject_id", "age_group"]
    cols = meta + [c for c in df.columns if c not in meta]
    return df[cols]
