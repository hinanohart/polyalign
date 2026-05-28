"""Matryoshka prefix-nested SAE alignment.

Implements the prefix-nested structure from Bussmann et al. 2025
(arXiv:2503.17547) for the polyalign cross-architecture setting:
for each prefix length p in [d/8, d/4, d/2, d], the first p features
of each SAE decoder are aligned via Sinkhorn-OT and the resulting
plans form a granularity hierarchy.

The polyalign-specific twist: prefixes are applied to FEATURE rows
(the SAE feature dictionary), not to d_model. This lets us inspect
coarse-grained features at p=d/8 and refine to fine-grained at p=d.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from polyalign._types import SAEBundle
from polyalign.alignment.sinkhorn import sinkhorn_align


@dataclass(frozen=True)
class MatryoshkaResult:
    """Result of a prefix-nested alignment between two SAE bundles."""

    prefix_lengths: tuple[int, ...]
    plans: tuple[np.ndarray, ...]


def default_prefix_schedule(n_features: int) -> tuple[int, ...]:
    """Return the canonical 4-stage prefix schedule [n/8, n/4, n/2, n].

    Each stage is clamped to a minimum of 2 features so that even tiny
    test bundles produce a non-degenerate Sinkhorn problem.
    """
    if n_features < 2:
        raise ValueError(f"need n_features >= 2 for Matryoshka, got {n_features}")
    candidate = [n_features // 8, n_features // 4, n_features // 2, n_features]
    out: list[int] = []
    for p in candidate:
        out.append(max(2, p))
    out_sorted = sorted(set(out))
    return tuple(out_sorted)


class MatryoshkaWrapper:
    """Wraps an SAEBundle with a 4-stage prefix mask.

    The wrapper exposes `prefix(p)` to materialize a new SAEBundle whose
    decoder is the first `p` rows of the wrapped bundle's decoder.
    """

    def __init__(self, bundle: SAEBundle, prefix_lengths: Sequence[int] | None = None):
        self.bundle = bundle
        if prefix_lengths is None:
            prefix_lengths = default_prefix_schedule(bundle.n_features)
        clamped = []
        for p in prefix_lengths:
            if p < 2:
                raise ValueError(f"prefix length {p} < 2 is degenerate")
            clamped.append(min(int(p), bundle.n_features))
        self.prefix_lengths: tuple[int, ...] = tuple(sorted(set(clamped)))

    def prefix(self, p: int) -> SAEBundle:
        if p < 2 or p > self.bundle.n_features:
            raise ValueError(f"prefix p={p} out of range [2, {self.bundle.n_features}]")
        return SAEBundle(
            model_id=f"{self.bundle.model_id}@p{p}",
            architecture=self.bundle.architecture,
            layer=self.bundle.layer,
            decoder=self.bundle.decoder[:p, :].copy(),
            encoder=(None if self.bundle.encoder is None else self.bundle.encoder[:, :p].copy()),
            feature_ids=(
                None if self.bundle.feature_ids is None else list(self.bundle.feature_ids[:p])
            ),
        )


def multi_granularity_alignment(
    bundle_a: SAEBundle,
    bundle_b: SAEBundle,
    prefix_lengths: Sequence[int] | None = None,
    *,
    reg: float = 0.05,
    cost: str = "cosine",
) -> MatryoshkaResult:
    """Run Sinkhorn-OT at each prefix length and return the family of plans."""
    if bundle_a.n_features != bundle_b.n_features:
        n_common = min(bundle_a.n_features, bundle_b.n_features)
    else:
        n_common = bundle_a.n_features

    wrapper_a = MatryoshkaWrapper(bundle_a, prefix_lengths)
    if prefix_lengths is None:
        schedule = default_prefix_schedule(n_common)
    else:
        schedule = wrapper_a.prefix_lengths
    wrapper_a = MatryoshkaWrapper(bundle_a, schedule)
    wrapper_b = MatryoshkaWrapper(bundle_b, schedule)

    plans: list[np.ndarray] = []
    for p in schedule:
        pa = wrapper_a.prefix(min(p, bundle_a.n_features))
        pb = wrapper_b.prefix(min(p, bundle_b.n_features))
        plans.append(sinkhorn_align(pa, pb, reg=reg, cost=cost))
    return MatryoshkaResult(prefix_lengths=schedule, plans=tuple(plans))


def prefix_recon_error(
    bundle: SAEBundle,
    activations: np.ndarray,
    prefix_lengths: Sequence[int] | None = None,
) -> dict[int, float]:
    """Compute per-prefix reconstruction error on synthetic activations.

    Used by `tests/test_matryoshka.py` to verify the Bussmann 2025
    monotonicity property: longer prefix -> lower recon error.

    activations shape: (n_samples, n_features). The reconstruction at
    prefix p uses only the first p columns of activations against the
    first p rows of decoder.
    """
    if bundle.encoder is None and activations.shape[1] != bundle.n_features:
        raise ValueError(
            f"activations columns {activations.shape[1]} != n_features {bundle.n_features}"
        )
    schedule = (
        default_prefix_schedule(bundle.n_features)
        if prefix_lengths is None
        else tuple(sorted({int(p) for p in prefix_lengths}))
    )
    target = activations @ bundle.decoder
    out: dict[int, float] = {}
    for p in schedule:
        recon = activations[:, :p] @ bundle.decoder[:p, :]
        err = float(np.linalg.norm(recon - target, "fro"))
        out[p] = err
    return out
