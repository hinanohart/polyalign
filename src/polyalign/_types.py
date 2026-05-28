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
from typing import Final, Literal

import numpy as np

Architecture = Literal["transformer", "ssm", "hybrid"]

MAMBA_SSM_HOOK_SITE: Final[str] = "out_proj_out"
"""recurrentlens convention for the Mamba/SSM hook site used during SAE feature
extraction. Recorded as a constant in v0.1.0a2 so that downstream code can
reference the agreed string without re-parsing docs. Live Mamba hook
extraction lands in v0.1.1 (see docs/CLAIM.md `Live model integration`)."""


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
    """Tuple of (model_i, model_j, feature_i, feature_j) edges associated with this
    vertex. For N>=3 in v0.1.0a2 the feature indices are chosen by a star projection
    from bundle 0 (feature_j = argmax of P_{0,j}[anchor]); edges (i,j) with i>0 are
    recorded for downstream reporting and not enforced during construction. Full
    pairwise-consistent clique enumeration is deferred to v0.2."""

    joint_probability: float
    """For N=2: P_{0,1}[feature_0, feature_1] / max(P_{0,1}.sum(), eps).
    For N>=3: product of edge probabilities (0,j) for j>0 from the star projection;
    NOT normalized to a probability and may take values outside [0, 1] (the
    `extract_polytope_vertices` test only asserts the N=2 path is in [0, 1])."""

    coverage_lower_bound: float
    """Split-conformal heuristic lower bound on coverage for this vertex. On the
    synthetic Gaussian decoders used in v0.1.0a2 the marginal quantile q approaches
    1.0 by construction (nonconformity = 1 - p with p ~ 1/(n_a*n_b)), so this
    value numerically approaches `joint_probability`. See docs/CLAIM.md."""

    cycle_consistent: bool
    """Output of the `_cycle_consistent` predicate. At the default
    `cycle_threshold=0.0` this is structurally True for any non-negative transport
    plan (the predicate `direct < 0 and composed < 0` cannot fire). A non-trivial
    threshold > 0 is required for the flag to discriminate; a positive default
    will land in v0.2."""


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
