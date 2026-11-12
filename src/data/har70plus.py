"""
HAR70+ dataset loader (Ustad et al., 2023).

18 older adults aged 70–95, two accelerometers (thigh + lower back), 50 Hz,
annotated with 7 ADL types. No fall class — used for RQ1 signal analysis and
false-alarm measurement on elderly ADLs.

Expected layout after download:
    data/raw/har70plus/
        subject_<id>/
            thigh.csv      (time, ax, ay, az, label)
            back.csv
        OR a single merged CSV from the Kaggle version.

The loader normalises to: (time_s, ax, ay, az, label) with units in g (50 Hz).
"""

from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_RATE = 50  # Hz

ACTIVITY_MAP = {
    1: "walking",
    2: "shuffling",
    3: "stairs_ascending",
    4: "stairs_descending",
    5: "standing",
    6: "sitting",
    7: "lying",
}


def load_har70plus(root: str | Path) -> pd.DataFrame:
    """
    Load HAR70+ data.

    Returns a DataFrame with columns:
        subject_id, time_s, ax, ay, az, activity_id, activity_label
    at 50 Hz with acceleration in g.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"HAR70+ root not found: {root}")

    frames = []

    # Try single-CSV Kaggle layout first
    for csv in sorted(root.glob("*.csv")):
        df = _load_flat_csv(csv)
        if df is not None:
            frames.append(df)
            break

    # Try per-subject subdirectory layout
    if not frames:
        for subj_dir in sorted(root.iterdir()):
            if not subj_dir.is_dir():
                continue
            subj_id = subj_dir.name
            for placement in ("thigh", "back"):
                f = subj_dir / f"{placement}.csv"
                if f.exists():
                    df = _load_per_subject_csv(f, subj_id, placement)
                    frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No recognisable HAR70+ files found under {root}. "
            "See data/README.md for expected structure."
        )

    out = pd.concat(frames, ignore_index=True)
    out["activity_label"] = out["activity_id"].map(ACTIVITY_MAP).fillna("unknown")
    return out


def _load_flat_csv(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"ax", "ay", "az"}
    if not required.issubset(df.columns):
        return None

    # Infer subject column
    subj_col = next((c for c in df.columns if "subject" in c or c == "id"), None)
    df["subject_id"] = df[subj_col].astype(str) if subj_col else "unknown"

    label_col = next((c for c in df.columns if "label" in c or "activity" in c), None)
    df["activity_id"] = df[label_col].astype(int) if label_col else -1

    time_col = next((c for c in df.columns if "time" in c), None)
    if time_col:
        df["time_s"] = df[time_col].astype(float)
    else:
        df["time_s"] = np.arange(len(df)) / SAMPLE_RATE

    return df[["subject_id", "time_s", "ax", "ay", "az", "activity_id"]].copy()


def _load_per_subject_csv(
    path: Path, subject_id: str, placement: str
) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] >= 5:
        df.columns = ["time_s", "ax", "ay", "az", "activity_id"] + [
            f"extra_{i}" for i in range(df.shape[1] - 5)
        ]
    elif df.shape[1] == 4:
        df.columns = ["ax", "ay", "az", "activity_id"]
        df["time_s"] = np.arange(len(df)) / SAMPLE_RATE
    else:
        raise ValueError(f"Unexpected column count in {path}: {df.shape[1]}")

    df["subject_id"] = f"{subject_id}_{placement}"
    df["activity_id"] = df["activity_id"].astype(int)
    return df[["subject_id", "time_s", "ax", "ay", "az", "activity_id"]].copy()
