# Changelog

All notable changes to **polyalign** are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and uses pre-release tags (e.g. `0.1.0a1`) per [PEP 440](https://peps.python.org/pep-0440/).

## [0.1.0a1] - 2026-05-29 (pre-alpha)

### Initial release

- `polyalign.SAEBundle` type: `model_id` + `architecture` (transformer/ssm/hybrid) + `layer` + decoder/encoder.
- `polyalign.alignment.sinkhorn.sinkhorn_align(bundle_a, bundle_b, reg, cost)`: Sinkhorn-OT pairwise alignment (POT backend).
- `polyalign.alignment.pairwise.pairwise_alignments(bundles)`: N x (N-1) / 2 pairwise alignments.
- `polyalign.matryoshka.MatryoshkaWrapper`: prefix-nested alignment over `[d/8, d/4, d/2, d]` 4 stages.
- `polyalign.conformal.otcp.otcp_calibrate(P, calib_pairs, alpha)`: split conformal coverage q (marginal mode default).
- `polyalign.calibration.pava`: vendored PAVA monotone isotonic regression from foldconsensus (Apache-2.0).
- `polyalign.calibration.ece`: equal-frequency-bin Expected Calibration Error.
- `polyalign.polytope.vertices.extract_polytope_vertices(...)`: novel polyalign core — Pareto-front top-k vertices with cycle-consistency.
- `polyalign-lint` CLI (typer).
- CI matrix: Python 3.10 / 3.11 / 3.12 on Linux.
- Dependabot security-updates only (version-updates limit 0, per `feedback_github_org_hygiene_2026-05-25`).
- Honest-marketing CI grep: ban list defined in `.github/banned-phrases.txt` rejects unverifiable comparative claims; the list itself is not duplicated in user-facing docs (avoids the self-reference trap in `feedback_ci-grep-bre-vs-ere-2026-05-24`).
- Tests >= 60, coverage >= 85%.

### Known limitations

- All `v0.1.0a1` ablation metrics are computed against **synthetic ground truth** (Case C activated at S0 due to live cross-model SAE inference being infeasible inside the build session). All synthetic feature pairs carry the `[DEMO]` prefix in `datasets/ground_truth_v0.1.0a1.jsonl`.
- Live cross-model SAE extraction (Llama-3 + Gemma-3 + Mamba-2) is **deferred to v0.1.1**.
- Poincare ball / hyperbolic cost matrix, OKLab visualization, and Muon optimizer are **deferred to v0.2**.
- Conditional-coverage OTCP per architecture-pair is implemented but the marginal mode is the default in this release. Switch via `otcp_calibrate(..., conditional=True)`.

### Provenance / what polyalign composes

- `gavagai` (MIT) — HookedSAEBundle reference + score pattern.
- `recurrentlens` (Apache-2.0) — SSM hook convention `out_proj_out`.
- `hybridlens` (Apache-2.0) — Hybrid SAE pattern reference.
- `foldgauge` (Apache-2.0) — split conformal reference.
- `foldconsensus` (Apache-2.0) — vendored PAVA + ECE pattern.
- `mosaicraft-active-vision` (Apache-2.0) — Sinkhorn-OT assignment pattern reference.
- POT (MIT) — Sinkhorn-Knopp solver.

See [NOTICE](NOTICE).
