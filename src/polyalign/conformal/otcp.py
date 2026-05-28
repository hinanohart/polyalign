"""OTCP - Optimal-Transport Conformal Prediction.

Reference: arXiv:2501.18991. polyalign uses split conformal calibration
on top of a Sinkhorn-OT transport plan to attach a coverage band q to
each candidate alignment vertex.

The default mode is **marginal**: q is a single quantile computed from
all (model_pair, feature_pair) calibration scores pooled together,
assuming exchangeability across architecture pairs.

The fallback mode is **conditional**: q is computed per architecture-pair
(transformer-transformer, transformer-ssm, ssm-hybrid, ...) when the
marginal exchangeability assumption is rejected (see CLAIM.md).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np

from polyalign._types import SAEBundle


class CalibPair(NamedTuple):
    """A single calibration sample for OTCP."""

    i: int
    j: int
    feature_a: int
    feature_b: int


def _scores_from_plan(plan: np.ndarray, pairs: Sequence[CalibPair]) -> np.ndarray:
    """Extract per-pair nonconformity scores 1 - p_{ij} from a transport plan."""
    if plan.ndim != 2:
        raise ValueError(f"plan must be 2D, got shape {plan.shape}")
    plan = plan / max(plan.sum(), 1e-12)  # normalize to a joint over (n_a, n_b)
    out = np.empty(len(pairs), dtype=np.float64)
    for k, pair in enumerate(pairs):
        if not (0 <= pair.feature_a < plan.shape[0]) or not (0 <= pair.feature_b < plan.shape[1]):
            raise IndexError(
                f"calibration pair (feature_a={pair.feature_a}, feature_b={pair.feature_b}) "
                f"out of plan shape {plan.shape}"
            )
        out[k] = 1.0 - float(plan[pair.feature_a, pair.feature_b])
    return out


def otcp_calibrate(
    plan: np.ndarray,
    calib_pairs: Sequence[CalibPair],
    *,
    alpha: float = 0.1,
) -> float:
    """Split-conformal marginal quantile on plan scores.

    Returns q such that with probability >= 1 - alpha over a fresh
    test pair drawn exchangeably from the same distribution,
    1 - plan[a, b] <= q.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if len(calib_pairs) == 0:
        raise ValueError("calib_pairs is empty")
    scores = _scores_from_plan(plan, calib_pairs)
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    sorted_scores = np.sort(scores)
    return float(sorted_scores[rank - 1])


def otcp_calibrate_conditional(
    plans: dict[tuple[int, int], np.ndarray],
    bundles: Sequence[SAEBundle],
    calib_pairs: Sequence[CalibPair],
    *,
    alpha: float = 0.1,
) -> dict[tuple[str, str], float]:
    """Per-architecture-pair conditional split-conformal quantile.

    Used as a fallback when marginal exchangeability is rejected
    (S0 OQ2 degrade). Returns a dict keyed by (arch_i, arch_j) with
    arch_i <= arch_j lexicographically.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    grouped: dict[tuple[str, str], list[float]] = {}
    for pair in calib_pairs:
        plan = plans.get((pair.i, pair.j))
        if plan is None:
            plan_rev = plans.get((pair.j, pair.i))
            if plan_rev is None:
                continue
            plan = plan_rev.T
        score = _scores_from_plan(plan, [pair])[0]
        arch_i = bundles[pair.i].architecture
        arch_j = bundles[pair.j].architecture
        a, b = sorted((str(arch_i), str(arch_j)))
        key: tuple[str, str] = (a, b)
        grouped.setdefault(key, []).append(float(score))

    out: dict[tuple[str, str], float] = {}
    for key, scores in grouped.items():
        n = len(scores)
        rank = int(np.ceil((n + 1) * (1.0 - alpha)))
        rank = min(max(rank, 1), n)
        sorted_scores = np.sort(np.asarray(scores, dtype=np.float64))
        out[key] = float(sorted_scores[rank - 1])
    return out


def coverage_lower_bound(
    plan_value: float,
    q: float,
    mode: Literal["marginal", "conditional"] = "marginal",
) -> float:
    """Convert OTCP quantile into a per-vertex coverage lower bound.

    Returns max(0, 1 - (1 - plan_value) / max(q, eps)) clipped to [0, 1].
    This is a heuristic conversion of nonconformity score to a
    [0, 1] coverage lower bound; honest scope is documented in CLAIM.md.
    """
    _ = mode  # marginal vs conditional decision is in the q value chosen
    eps = 1e-12
    nonconformity = 1.0 - plan_value
    ratio = nonconformity / max(q, eps)
    lb = 1.0 - ratio
    return float(np.clip(lb, 0.0, 1.0))
