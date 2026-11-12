"""
SisFall dataset loader.

SisFall (Sucerquia et al., 2017) contains tri-axial accelerometer (×2) and gyroscope
signals at ~200 Hz from 23 young adults (SA, 19–30 y) and 15 elderly (SE, 60–75 y).
Young subjects performed both falls (F01–F15) and ADLs (D01–D19); elderly performed
ADLs only (one elderly subject also performed falls — confirmed in audit).

File naming convention:  <ActivityCode><SubjectCode><RepetitionCode>.txt
  e.g.  F01SA01R01.txt  = Fall type 01, Subject Adult 01, Repetition 01
        D05SE03R02.txt  = ADL type 05, Subject Elderly 03, Repetition 02

Each file: 6 columns — ADXL345 (ax,ay,az), ITG-3200 gyro (gx,gy,gz), MMA8451Q (ax2,ay2,az2)
Actually 9 columns in some versions: ax1 ay1 az1 gx gy gz ax2 ay2 az2
Raw units from sensor registers; convert using documented scale factors (see constants below).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# SisFall sensor scale factors (from dataset documentation)
ADXL345_SCALE = 1 / 256        # ±16 g range → g
ITG3200_SCALE = 1 / 14.375     # °/s per LSB
MMA8451_SCALE = 1 / 4096       # ±8 g range → g

SAMPLE_RATE = 200  # Hz

FALL_CODES = {f"F{i:02d}" for i in range(1, 16)}    # F01–F15
ADL_CODES  = {f"D{i:02d}" for i in range(1, 20)}    # D01–D19

YOUNG_PREFIX   = "SA"
ELDERLY_PREFIX = "SE"


@dataclass
class Trial:
    subject_id: str          # e.g. "SA01", "SE03"
    activity:   str          # e.g. "F01", "D05"
    repetition: int
    is_fall:    bool
    age_group:  str          # "young" | "elderly"
    # shape (N, 9): ax1 ay1 az1 gx gy gz ax2 ay2 az2  — calibrated units (g, °/s)
    signal:     np.ndarray = field(repr=False)

    @property
    def duration_s(self) -> float:
        return len(self.signal) / SAMPLE_RATE


_FILENAME_RE = re.compile(
    r"^([FD]\d{2})_?(S[AE]\d+)_?(R\d+)\.txt$", re.IGNORECASE
)


def _parse_filename(name: str):
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    activity, subject, rep = m.group(1).upper(), m.group(2).upper(), m.group(3)
    return activity, subject, int(rep[1:])


def _read_trial(path: Path) -> np.ndarray:
    """Read one SisFall .txt file → (N, 9) calibrated float array."""
    # Lines end with ';' — strip it before parsing
    lines = path.read_text().splitlines()
    cleaned = [l.rstrip("; \t") for l in lines if l.strip()]
    from io import StringIO
    raw = pd.read_csv(StringIO("\n".join(cleaned)), header=None, sep=",", dtype=float).values
    if raw.shape[1] < 9:
        raise ValueError(f"{path.name}: expected 9 columns, got {raw.shape[1]}")
    signal = raw[:, :9].copy()
    signal[:, 0:3] *= ADXL345_SCALE   # ax1,ay1,az1 → g
    signal[:, 3:6] *= ITG3200_SCALE   # gx,gy,gz   → °/s
    signal[:, 6:9] *= MMA8451_SCALE   # ax2,ay2,az2 → g
    return signal


def load_sisfall(
    root: str | Path,
    *,
    sensors: str = "adxl345",   # "adxl345" | "mma8451" | "both" | "all"
    include_elderly_falls: bool = True,
) -> list[Trial]:
    """
    Load all SisFall trials from *root* directory.

    Parameters
    ----------
    root:
        Path to the SisFall root directory containing SA*/SE* subject folders.
    sensors:
        Which accelerometer(s) to keep in the returned signal:
        - "adxl345": columns 0–2 (+ gyro 3–5)  [default — waist-mount primary sensor]
        - "mma8451": columns 6–8 (+ gyro 3–5)
        - "both":    columns 0–8
        - "all":     alias for "both"
    include_elderly_falls:
        If False, drop the single elderly subject who performed scripted falls.

    Returns
    -------
    List of :class:`Trial` objects, one per .txt file.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"SisFall root not found: {root}")

    _sensor_cols = {
        "adxl345": list(range(6)),       # ax1 ay1 az1 gx gy gz
        "mma8451": [6, 7, 8, 3, 4, 5],  # ax2 ay2 az2 gx gy gz
        "both":    list(range(9)),
        "all":     list(range(9)),
    }
    if sensors not in _sensor_cols:
        raise ValueError(f"sensors must be one of {list(_sensor_cols)}; got {sensors!r}")
    cols = _sensor_cols[sensors]

    trials = []
    for txt in sorted(root.rglob("*.txt")):
        parsed = _parse_filename(txt.name)
        if parsed is None:
            continue
        activity, subject, rep = parsed

        if activity not in FALL_CODES and activity not in ADL_CODES:
            continue

        age_group = "elderly" if subject.startswith(ELDERLY_PREFIX) else "young"
        is_fall   = activity in FALL_CODES

        if not include_elderly_falls and age_group == "elderly" and is_fall:
            continue

        try:
            signal = _read_trial(txt)[:, cols]
        except (ValueError, pd.errors.ParserError) as exc:
            print(f"[warn] skipping {txt.name}: {exc}")
            continue

        trials.append(Trial(
            subject_id=subject,
            activity=activity,
            repetition=rep,
            is_fall=is_fall,
            age_group=age_group,
            signal=signal,
        ))

    return trials


def trials_to_dataframe(trials: list[Trial]) -> pd.DataFrame:
    """Return a tidy metadata DataFrame (no signals) for quick inspection."""
    rows = []
    for t in trials:
        rows.append({
            "subject_id": t.subject_id,
            "activity":   t.activity,
            "repetition": t.repetition,
            "is_fall":    t.is_fall,
            "age_group":  t.age_group,
            "n_samples":  len(t.signal),
            "duration_s": t.duration_s,
        })
    return pd.DataFrame(rows)
