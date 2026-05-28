"""polyalign: N-model M-architecture SAE alignment polytope.

Sinkhorn-OT pairwise x Matryoshka prefix x OTCP split conformal coverage.
"""

from polyalign._types import (
    AlignmentPolytope,
    CoverageReport,
    ParetoFront,
    SAEBundle,
    Vertex,
)
from polyalign._version import __version__
from polyalign.polytope.vertices import alignment_polytope

__all__ = [
    "AlignmentPolytope",
    "CoverageReport",
    "ParetoFront",
    "SAEBundle",
    "Vertex",
    "__version__",
    "alignment_polytope",
]
