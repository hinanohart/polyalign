"""Pairwise Sinkhorn-OT alignment matrices over N SAE bundles.

For N bundles, this returns N*(N-1)/2 plans keyed by (i, j) with i < j.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from polyalign._types import SAEBundle
from polyalign.alignment.sinkhorn import sinkhorn_align


def pairwise_alignments(
    bundles: Sequence[SAEBundle],
    *,
    reg: float = 0.05,
    cost: str = "cosine",
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> dict[tuple[int, int], np.ndarray]:
    """Compute all unordered pairs (i, j) with i < j.

    Each plan P_{i,j} has shape (bundles[i].n_features, bundles[j].n_features).
    """
    n = len(bundles)
    if n < 2:
        raise ValueError(f"pairwise_alignments needs >= 2 bundles, got {n}")

    plans: dict[tuple[int, int], np.ndarray] = {}
    for i in range(n):
        for j in range(i + 1, n):
            plans[(i, j)] = sinkhorn_align(
                bundles[i],
                bundles[j],
                reg=reg,
                cost=cost,
                max_iter=max_iter,
                tol=tol,
            )
    return plans
