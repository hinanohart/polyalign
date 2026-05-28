# polyalign

[![status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](https://github.com/hinanohart/polyalign/releases)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)

> **N-model M-architecture SAE alignment polytope** — Sinkhorn-OT pairwise x Matryoshka prefix x OTCP split conformal coverage.

`polyalign` aligns Sparse Autoencoder (SAE) feature dictionaries across **N >= 2 models** and **M >= 1 architectures** (Transformer / SSM / Hybrid), and returns an **alignment polytope** — a Pareto-front of top-k vertices, each with a split-conformal coverage band and cycle-consistency check.

## Status

> `v0.1.0a1` is a **pre-alpha** release. See [docs/CLAIM.md](docs/CLAIM.md) for the explicit `[CLAIM]` vs `[non-CLAIM]` boundary. **All ablation metrics in this release are computed against synthetic ground truth** (`[DEMO]`-prefixed feature pairs); real cross-model concept pair curation is deferred to `v0.1.1`.

## Install

```bash
pip install polyalign
# or with torch + transformers for live model SAE extraction (deferred to v0.1.1):
pip install 'polyalign[torch,llama3,saelens]'
```

## Quickstart

```python
import numpy as np
from polyalign import SAEBundle, alignment_polytope

# Two pre-trained SAE decoders (n_features x d_model)
bundle_a = SAEBundle(
    model_id="gpt2-small",
    architecture="transformer",
    layer=6,
    decoder=np.random.RandomState(0).randn(64, 32).astype(np.float32),
)
bundle_b = SAEBundle(
    model_id="pythia-160m",
    architecture="transformer",
    layer=6,
    decoder=np.random.RandomState(1).randn(64, 32).astype(np.float32),
)

result = alignment_polytope([bundle_a, bundle_b], top_k=5)
print(f"vertices: {len(result.vertices)}")
for v in result.vertices:
    print(f"  joint_p={v.joint_probability:.4f}  coverage_lb={v.coverage_lower_bound:.3f}")
```

## CLI

```bash
polyalign-lint --help
polyalign-lint align --bundles bundle_a.npz,bundle_b.npz --top-k 5
```

## What polyalign computes

| Layer | Component | Source |
|-------|-----------|--------|
| Input | N >= 2 SAE bundles (Transformer / SSM / Hybrid) | gavagai HookedSAEBundle |
| Core  | Sinkhorn-OT pairwise alignments | POT (MIT) + cross-arch cost matrix |
| Core  | Matryoshka prefix nested alignment (4 stages) | Bussmann et al. 2025 reproduction |
| Calib | OTCP split conformal coverage q | foldgauge / arXiv 2501.18991 |
| Calib | PAVA monotone isotonic | foldconsensus (vendored copy) |
| Out   | Alignment polytope vertices + Pareto front | polyalign novel core |

## Related work

`polyalign` is **not** a replacement for any single existing tool — it composes 11 techniques across 7 genres (mechanistic interpretability / AI safety / cross-architecture eval / calibration / model diffing / distributed agents / SSM ecosystem). Differentiation from the most closely related open-source and academic work:

- **`gavagai`** (hinanohart, MIT, v0.2.1) — pairwise SAE indeterminacy score. `polyalign` extends to **N x M polytope vertices** (output: ranked vertex set with coverage band and Pareto front, not a scalar).
- **`AlignSAE`** (arXiv 2512.02004) — concept-aligned SAE training (training-time). `polyalign` is **post-hoc alignment** over pre-trained SAEs (no retraining required).
- **Anthropic `crosscoder`** (closed-source) — 2-model differential SAE. `polyalign` is **open-source N-model multi-architecture** and ships an OTCP split-conformal coverage band per vertex.
- **`SPARC`** (arXiv 2507.06265) — concept-aligned SAEs for cross-model / cross-modal training. `polyalign` is post-hoc and does not retrain SAEs.
- **`ckkissane/crosscoder-model-diff-replication`** — Anthropic crosscoder Euclidean replication. `polyalign` ships OTCP coverage + Sinkhorn-OT + first-class SSM carrier support.
- **OpenMOSS / Llamascopium** — per-model Matryoshka SAE. `polyalign` ships **cross-architecture** Matryoshka with Sinkhorn-OT pairwise alignment.

## Honest-marketing scope

polyalign v0.1.0a1 ships:

- Sinkhorn-OT pairwise alignment (live, tested)
- Matryoshka prefix-nested alignment 4-stage (live, tested)
- OTCP split conformal coverage (live, tested, marginal mode default)
- Alignment polytope vertex extraction + Pareto front (live, tested, novel core)
- CLI smoke test (live)

polyalign v0.1.0a1 does **not** ship:

- live cross-model SAE extraction at scale (Llama-3 + Gemma-3 + Mamba-2) — deferred to v0.1.1
- real hand-labeled ground truth pairs — synthetic `[DEMO]` only in this release
- Poincare ball / hyperbolic cost matrix — deferred to v0.2
- OKLab perceptually-uniform visualization — deferred to v0.2
- Muon optimizer integration — deferred to v0.2+

See [docs/CLAIM.md](docs/CLAIM.md), [CHANGELOG.md](CHANGELOG.md) for the full scope.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

```bibtex
@software{polyalign_2026,
  author = {hinanohart},
  title  = {polyalign: N-model M-architecture SAE alignment polytope},
  year   = {2026},
  url    = {https://github.com/hinanohart/polyalign},
  version = {0.1.0a1},
  license = {Apache-2.0},
}
```

## Reproducibility

<!-- ABLATION:BEGIN -->

### Ablation (synthetic, `mode=synthetic`, dataset_n=36)

> All values reproduced from `results/v0.1.0a1_ablation.json` (seed=0,
> Python 3.12.3 on Linux, polyalign 0.1.0a1).
> `[DEMO]` prefix is mechanically enforced on every ground-truth row (Case C (S0 OQ3 degrade)).
> These numbers reflect **synthetic alignment quality on random Gaussian decoders**, not real cross-model concept matching - see [docs/CLAIM.md](docs/CLAIM.md).

| setting | top1 | top5 | ece | otcp_coverage | n_vertices | cycle_consistency_rate |
|---------|------|------|-----|---------------|------------|------------------------|
| `cosine_r0.05_transformer-only_random` | 0.000 | 0.000 | 0.9992 | 1.0000 | 5 | 1.000 |
| `cosine_r0.05_transformer-only_planted` | 0.042 | 0.208 | 0.9983 | 1.0000 | 5 | 1.000 |
| `cosine_r0.05_ssm-only_random` | 0.000 | 0.000 | 0.9992 | 1.0000 | 5 | 1.000 |
| `cosine_r0.05_ssm-only_planted` | 0.042 | 0.208 | 0.9983 | 1.0000 | 5 | 1.000 |
| `cosine_r0.05_hybrid-only_random` | 0.000 | 0.000 | 0.9992 | 1.0000 | 5 | 1.000 |
| `cosine_r0.05_hybrid-only_planted` | 0.042 | 0.208 | 0.9983 | 1.0000 | 5 | 1.000 |
| `cosine_r0.05_mixed_random` | 0.000 | 0.000 | 0.9992 | 1.0000 | 5 | 1.000 |
| `cosine_r0.05_mixed_planted` | 0.042 | 0.208 | 0.9983 | 1.0000 | 5 | 1.000 |
| `l2_r0.05_mixed_random` | 0.000 | 0.000 | 0.9992 | 1.0000 | 5 | 1.000 |
| `l2_r0.05_mixed_planted` | 0.042 | 0.208 | 0.9983 | 1.0000 | 5 | 1.000 |
| `cosine_r0.01_mixed_random` | 0.000 | 0.000 | 0.9983 | 1.0000 | 5 | 1.000 |
| `cosine_r0.01_mixed_planted` | 0.042 | 0.208 | 0.9983 | 1.0000 | 5 | 1.000 |
| `cosine_r0.1_mixed_random` | 0.000 | 0.000 | 0.9997 | 1.0000 | 5 | 1.000 |
| `cosine_r0.1_mixed_planted` | 0.042 | 0.208 | 0.9988 | 1.0000 | 5 | 1.000 |

> Reproduce: `uv run python scripts/run_ablation.py --seed 0`

<!-- ABLATION:END -->
