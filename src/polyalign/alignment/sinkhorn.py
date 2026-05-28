"""Sinkhorn-OT entropic optimal transport between two SAE feature dictionaries.

Following Cuturi 2013 + POT (https://pythonot.github.io/) `ot.sinkhorn`,
this module wraps the iterative Sinkhorn-Knopp updates with a polyalign-
specific cost matrix builder.

Supported cost matrices:

- "cosine"  : 1 - normalized inner product between decoder rows (default)
- "l2"      : pairwise Euclidean distance
- "poincare": v0.2 stub - currently fall back to "cosine" with a warning

Cross-architecture handling: when the two bundles have different
`architecture` values (e.g. transformer vs ssm), the cost matrix is
mean-normalized per row so that the OT regularization does not get
dominated by absolute distance scale differences.
"""

from __future__ import annotations

import warnings

import numpy as np
import ot

from polyalign._types import SAEBundle


def build_cost_matrix(
    bundle_a: SAEBundle,
    bundle_b: SAEBundle,
    cost: str = "cosine",
) -> np.ndarray:
    """Build a cost matrix C of shape (n_a, n_b) between two SAE feature dictionaries.

    Parameters
    ----------
    bundle_a, bundle_b
        polyalign SAEBundle instances.
    cost
        One of "cosine" (default), "l2", "poincare" (v0.2 stub).

    Returns
    -------
    C : ndarray of shape (n_a, n_b), dtype float64
    """
    if bundle_a.d_model != bundle_b.d_model:
        raise ValueError(
            f"d_model mismatch: bundle_a.d_model={bundle_a.d_model} vs "
            f"bundle_b.d_model={bundle_b.d_model}; cannot build cost matrix"
        )

    a = bundle_a.decoder.astype(np.float64, copy=False)
    b = bundle_b.decoder.astype(np.float64, copy=False)

    if cost == "cosine":
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        cm = 1.0 - a_norm @ b_norm.T
    elif cost == "l2":
        cm = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    elif cost == "poincare":
        warnings.warn(
            "cost='poincare' is a v0.2 stub; falling back to cosine. "
            "Track v0.2 backlog in CHANGELOG.md.",
            stacklevel=2,
        )
        return build_cost_matrix(bundle_a, bundle_b, cost="cosine")
    else:
        raise ValueError(f"unknown cost: {cost!r} (expected cosine|l2|poincare)")

    if bundle_a.architecture != bundle_b.architecture:
        row_means = cm.mean(axis=1, keepdims=True)
        col_means = cm.mean(axis=0, keepdims=True)
        cm = cm - 0.5 * row_means - 0.5 * col_means + cm.mean()

    cm = cm - cm.min()
    max_v = cm.max()
    if max_v > 0:
        cm = cm / max_v
    return cm


def sinkhorn_align(
    bundle_a: SAEBundle,
    bundle_b: SAEBundle,
    *,
    reg: float = 0.05,
    cost: str = "cosine",
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Compute a Sinkhorn-OT transport plan P of shape (n_a, n_b).

    Marginals are uniform on both sides (a row sum = 1/n_a per row).
    POT's `ot.sinkhorn` with method `sinkhorn_stabilized` is used for
    numerical stability at small reg.
    """
    if reg <= 0:
        raise ValueError(f"reg must be > 0, got {reg}")

    cm = build_cost_matrix(bundle_a, bundle_b, cost=cost)
    n_a, n_b = cm.shape
    a_marg = np.full(n_a, 1.0 / n_a)
    b_marg = np.full(n_b, 1.0 / n_b)

    plan = ot.sinkhorn(
        a_marg,
        b_marg,
        cm,
        reg=reg,
        method="sinkhorn_stabilized",
        numItermax=max_iter,
        stopThr=tol,
    )
    plan_arr = np.asarray(plan, dtype=np.float64)
    if not np.all(np.isfinite(plan_arr)):
        raise RuntimeError(
            f"Sinkhorn did not converge to a finite plan "
            f"(reg={reg}, max_iter={max_iter}); plan contains NaN/Inf. "
            "Try a larger reg or higher max_iter."
        )
    return plan_arr
