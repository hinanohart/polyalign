"""polyalign core dataclasses.

`SAEBundle` is the polyalign internal type wrapping a single SAE
(encoder / decoder pair) on a single model layer. The bundle records
the model id, architecture family (transformer / ssm / hybrid), and
layer index so cross-architecture alignment can apply the right cost
function and hook convention (`out_proj_out` for SSM via recurrentlens).

`Vertex` is one element of the alignment polytope: a sequence of
(model_i, model_j, feature_i, feature_j) edges that all correspond
to the same conceptual feature across N models, together with the
joint probability under the Sinkhorn-OT transport plans and a
split-conformal coverage lower bound.

`ParetoFront` and `AlignmentPolytope` package a top-k ordered set of
Vertices and the per-vertex coverage report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Architecture = Literal["transformer", "ssm", "hybrid"]


@dataclass(frozen=True)
class SAEBundle:
    """A pre-trained SAE on one layer of one model.

    decoder shape: (n_features, d_model). encoder is optional and, when
    present, has shape (d_model, n_features). polyalign treats the decoder
    as the canonical "feature directions" matrix, following gavagai's
    convention.
    """

    model_id: str
    architecture: Architecture
    layer: int
    decoder: np.ndarray
    encoder: np.ndarray | None = None
    feature_ids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.decoder.ndim != 2:
            raise ValueError(
                f"decoder must be 2D (n_features, d_model), got shape {self.decoder.shape}"
            )
        if self.architecture not in ("transformer", "ssm", "hybrid"):
            raise ValueError(
                f"architecture must be transformer | ssm | hybrid, got {self.architecture!r}"
            )
        if self.encoder is not None:
            n_feat, d_model = self.decoder.shape
            if self.encoder.shape != (d_model, n_feat):
                raise ValueError(
                    f"encoder shape {self.encoder.shape} inconsistent with "
                    f"decoder shape {self.decoder.shape}: expected ({d_model}, {n_feat})"
                )

    @property
    def n_features(self) -> int:
        return int(self.decoder.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.decoder.shape[1])


@dataclass(frozen=True)
class Vertex:
    """One alignment polytope vertex (a conceptually-aligned feature group)."""

    model_pairs: tuple[tuple[int, int, int, int], ...]
    """Tuple of (model_i, model_j, feature_i, feature_j) edges making this vertex."""

    joint_probability: float
    """Product of Sinkhorn transport probabilities for all edges (normalized)."""

    coverage_lower_bound: float
    """Split-conformal lower bound on coverage probability for this vertex."""

    cycle_consistent: bool
    """True iff for every triangle (A, B, C), A->B + B->C is consistent with A->C."""


@dataclass(frozen=True)
class CoverageReport:
    """OTCP coverage summary returned alongside the polytope."""

    alpha: float
    """User-requested mis-coverage rate."""

    quantile: float
    """Threshold q at which calibration set reaches 1 - alpha coverage."""

    n_calib: int
    """Number of calibration pairs used."""

    mode: Literal["marginal", "conditional"]
    """marginal (default) or per-architecture-pair conditional coverage."""


@dataclass(frozen=True)
class ParetoFront:
    """Pareto-dominant vertices ordered by joint_probability x coverage_lower_bound."""

    vertices: tuple[Vertex, ...]


@dataclass
class AlignmentPolytope:
    """Top-level polyalign result: vertices + Pareto front + coverage report."""

    vertices: list[Vertex]
    pareto_front: ParetoFront
    coverage: CoverageReport
    bundles: list[SAEBundle] = field(default_factory=list)

    @property
    def top_k(self) -> int:
        return len(self.vertices)
