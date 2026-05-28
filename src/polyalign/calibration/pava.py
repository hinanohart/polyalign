"""PAVA monotone isotonic regression.

Vendored from `foldconsensus.core.calibration` (Apache License 2.0,
github.com/hinanohart/foldconsensus). polyalign re-publishes a local
copy here to avoid a hard dependency on foldconsensus during
pre-alpha; see NOTICE for attribution.

The Pool-Adjacent-Violators (PAVA) algorithm projects an arbitrary
sequence onto the cone of non-decreasing sequences with respect to
weighted least-squares loss. We expose a numpy-only version with
optional weights.
"""

from __future__ import annotations

import numpy as np


def pava_monotone(y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Pool-Adjacent-Violators isotonic regression on `y`.

    Parameters
    ----------
    y
        1D array of values.
    weights
        Optional 1D array of positive weights, same length as `y`.

    Returns
    -------
    out : ndarray
        Non-decreasing isotonic projection of `y`.
    """
    y_arr = np.asarray(y, dtype=np.float64).copy()
    n = y_arr.shape[0]
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y_arr.shape}")
    if n == 0:
        return y_arr

    if weights is None:
        w_arr = np.ones(n, dtype=np.float64)
    else:
        w_arr = np.asarray(weights, dtype=np.float64).copy()
        if w_arr.shape != y_arr.shape:
            raise ValueError(f"weights shape {w_arr.shape} != y shape {y_arr.shape}")
        if np.any(w_arr <= 0):
            raise ValueError("weights must be positive")

    # block representation: (value, weight, start_index, end_index)
    values = y_arr.copy()
    block_w = w_arr.copy()
    block_start = np.arange(n)
    block_end = np.arange(n)
    n_blocks = n

    i = 0
    while i < n_blocks - 1:
        if values[i] <= values[i + 1]:
            i += 1
            continue
        # merge block i and i+1
        w_new = block_w[i] + block_w[i + 1]
        v_new = (block_w[i] * values[i] + block_w[i + 1] * values[i + 1]) / w_new
        values[i] = v_new
        block_w[i] = w_new
        block_end[i] = block_end[i + 1]
        # shift down
        values[i + 1 : n_blocks - 1] = values[i + 2 : n_blocks]
        block_w[i + 1 : n_blocks - 1] = block_w[i + 2 : n_blocks]
        block_start[i + 1 : n_blocks - 1] = block_start[i + 2 : n_blocks]
        block_end[i + 1 : n_blocks - 1] = block_end[i + 2 : n_blocks]
        n_blocks -= 1
        if i > 0:
            i -= 1

    out = np.empty(n, dtype=np.float64)
    for k in range(n_blocks):
        out[block_start[k] : block_end[k] + 1] = values[k]
    return out
