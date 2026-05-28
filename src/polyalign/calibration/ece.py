"""Equal-frequency Expected Calibration Error.

Implementation follows foldconsensus's equal-frequency binning pattern
(Apache-2.0): probabilities are sorted, partitioned into B equal-count
bins, and ECE is the absolute gap between mean predicted and mean
empirical per-bin accuracy weighted by bin mass.
"""

from __future__ import annotations

import numpy as np


def expected_calibration_error(probs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Return equal-frequency ECE in [0, 1].

    Parameters
    ----------
    probs
        1D array of predicted probabilities in [0, 1].
    correct
        1D array of 0/1 (or bool) ground-truth correctness.
    n_bins
        Number of equal-count bins (default 10).
    """
    p = np.asarray(probs, dtype=np.float64)
    c = np.asarray(correct, dtype=np.float64)
    if p.ndim != 1 or c.ndim != 1:
        raise ValueError(f"probs/correct must be 1D, got {p.shape} / {c.shape}")
    if p.shape != c.shape:
        raise ValueError(f"probs/correct shape mismatch: {p.shape} vs {c.shape}")
    n = p.shape[0]
    if n == 0:
        return 0.0
    if not (1 <= n_bins <= n):
        n_bins = max(1, min(n_bins, n))

    order = np.argsort(p)
    p_sorted = p[order]
    c_sorted = c[order]
    splits = np.array_split(np.arange(n), n_bins)
    ece = 0.0
    for idx in splits:
        if len(idx) == 0:
            continue
        bin_p = p_sorted[idx].mean()
        bin_c = c_sorted[idx].mean()
        ece += (len(idx) / n) * abs(bin_p - bin_c)
    return float(ece)
