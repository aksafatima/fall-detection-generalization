"""
Threshold-based fall detector.

Follows Bourke, O'Brien & Lyons (2007) and Kangas et al. (2008):
  1. Compute SVM of the accelerometer signal.
  2. FALL if SVM exceeds an impact threshold AND
     the post-impact SVM falls below a low-activity threshold.

The detector operates on a full trial signal (not windowed), so it is used in
P1/P2/P3 at the trial level before comparing against window-based ML models.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ThresholdDetector:
    """
    Parameters
    ----------
    impact_threshold_g : float
        SVM value (in g) above which an impact is flagged.
        Young-adult default ≈ 3–4 g; can be tuned downward for elderly data.
    inactivity_threshold_g : float
        SVM value below which a post-impact inactivity phase is detected.
    impact_window_s : float
        Duration before/after the peak to search for impact (seconds).
    inactivity_window_s : float
        Duration after the impact window to check for inactivity (seconds).
    fs : int
        Sampling rate of the input signal in Hz.
    """
    impact_threshold_g:    float = 3.0
    inactivity_threshold_g: float = 1.2
    impact_window_s:        float = 0.5
    inactivity_window_s:    float = 1.5
    fs:                     int   = 50

    def predict_trial(self, signal: np.ndarray) -> int:
        """
        Classify a single trial signal.

        Parameters
        ----------
        signal : (N, C) or (N,)
            If multi-channel, columns 0–2 are assumed to be ax, ay, az.

        Returns
        -------
        1 if fall detected, 0 otherwise.
        """
        if signal.ndim > 1:
            acc = signal[:, :3]
        else:
            acc = signal.reshape(-1, 1)
        svm = np.linalg.norm(acc, axis=1) if acc.shape[1] >= 3 else acc.flatten()

        impact_samples    = int(self.impact_window_s    * self.fs)
        inactivity_samples = int(self.inactivity_window_s * self.fs)

        impact_indices = np.where(svm > self.impact_threshold_g)[0]
        if len(impact_indices) == 0:
            return 0

        for idx in impact_indices:
            start = idx
            end   = min(idx + impact_samples + inactivity_samples, len(svm))
            post_impact = svm[start : end]
            if np.any(post_impact < self.inactivity_threshold_g):
                return 1

        return 0

    def predict(self, trials) -> np.ndarray:
        """
        Classify a list of Trial objects.

        Returns (N,) int array of predictions.
        """
        return np.array([self.predict_trial(t.signal) for t in trials])

    def tune_threshold(
        self,
        trials,
        impact_values: list[float] | None = None,
        inactivity_values: list[float] | None = None,
    ) -> tuple[float, float, float]:
        """
        Grid-search impact and inactivity thresholds on a provided trial list.

        Returns
        -------
        (best_impact, best_inactivity, best_f1)
        """
        from sklearn.metrics import f1_score

        if impact_values is None:
            impact_values = np.arange(1.5, 5.0, 0.25).tolist()
        if inactivity_values is None:
            inactivity_values = np.arange(0.8, 1.6, 0.1).tolist()

        y_true = np.array([int(t.is_fall) for t in trials])
        best_f1, best_imp, best_ina = -1, self.impact_threshold_g, self.inactivity_threshold_g

        for imp in impact_values:
            for ina in inactivity_values:
                self.impact_threshold_g    = imp
                self.inactivity_threshold_g = ina
                y_pred = self.predict(trials)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_imp, best_ina = f1, imp, ina

        self.impact_threshold_g    = best_imp
        self.inactivity_threshold_g = best_ina
        return best_imp, best_ina, best_f1
