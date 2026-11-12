"""
Harmonization utilities for cross-dataset experiments.

Different datasets use different sampling rates, sensor placements and unit scales.
This module provides resampling, magnitude computation and windowing so all datasets
are on a common footing before feature extraction or deep-model input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd


TARGET_FS = 50  # Hz — common denominator for SisFall (200 Hz) and HAR70+ (50 Hz)
WINDOW_SIZE_S = 2.56   # seconds — standard HAR window
OVERLAP = 0.5          # 50 % overlap


def resample_signal(signal: np.ndarray, orig_fs: int, target_fs: int = TARGET_FS) -> np.ndarray:
    """Polyphase resample from orig_fs to target_fs."""
    if orig_fs == target_fs:
        return signal
    g = gcd(orig_fs, target_fs)
    up, down = target_fs // g, orig_fs // g
    return resample_poly(signal, up, down, axis=0)


def lowpass_filter(
    signal: np.ndarray, cutoff_hz: float = 20.0, fs: int = TARGET_FS, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter (applied along axis 0)."""
    b, a = butter(order, cutoff_hz / (fs / 2), btype="low")
    return filtfilt(b, a, signal, axis=0)


def compute_svm(acc: np.ndarray) -> np.ndarray:
    """
    Signal Vector Magnitude (SVM) = ||[ax, ay, az]|| — gravity-inclusive.
    acc shape: (N, 3)  →  returns (N,)
    """
    return np.linalg.norm(acc, axis=1)


def compute_net_acceleration(acc: np.ndarray, g: float = 1.0) -> np.ndarray:
    """
    Net acceleration = SVM − g  (removes steady-state gravity contribution).
    Returns (N,) in the same units as acc (expected: g).
    """
    return compute_svm(acc) - g


def sliding_windows(
    signal: np.ndarray,
    label: int,
    fs: int = TARGET_FS,
    window_s: float = WINDOW_SIZE_S,
    overlap: float = OVERLAP,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment a trial into overlapping fixed-length windows.

    Returns
    -------
    windows : (n_windows, window_len, n_channels)
    labels  : (n_windows,)  — same label for every window of this trial
    """
    win_len  = int(window_s * fs)
    step     = int(win_len * (1 - overlap))
    n_channels = signal.shape[1] if signal.ndim > 1 else 1

    windows = []
    start = 0
    while start + win_len <= len(signal):
        w = signal[start : start + win_len]
        windows.append(w)
        start += step

    if not windows:
        return np.empty((0, win_len, n_channels)), np.empty(0, dtype=int)

    return np.stack(windows), np.full(len(windows), label, dtype=int)


def harmonize_datasets(
    sisfall_trials,
    *,
    sisfall_fs: int = 200,
    target_fs:  int = TARGET_FS,
    acc_cols: list[int] | None = None,
    window_s: float = WINDOW_SIZE_S,
    overlap:  float = OVERLAP,
    lowpass:  bool  = True,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Convert SisFall Trial objects into windowed arrays ready for modelling.

    Returns
    -------
    X        : (N, window_len, n_channels)
    y        : (N,)  binary fall labels
    subjects : (N,)  subject IDs — used for grouped CV splits
    groups   : (N,)  age group labels ("young" | "elderly")
    """
    if acc_cols is None:
        acc_cols = [0, 1, 2]  # ADXL345 x,y,z by default

    all_X, all_y, all_subj, all_grp = [], [], [], []

    for trial in sisfall_trials:
        sig = trial.signal[:, acc_cols].astype(np.float32)
        sig = resample_signal(sig, sisfall_fs, target_fs)
        if lowpass:
            sig = lowpass_filter(sig, fs=target_fs).astype(np.float32)

        label = int(trial.is_fall)
        wins, labels = sliding_windows(sig, label, fs=target_fs,
                                        window_s=window_s, overlap=overlap)
        if len(wins) == 0:
            continue

        all_X.append(wins)
        all_y.append(labels)
        all_subj.extend([trial.subject_id] * len(wins))
        all_grp.extend([trial.age_group] * len(wins))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y, all_subj, all_grp
