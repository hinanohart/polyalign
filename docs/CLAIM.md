# CLAIM.md — what polyalign claims and what it does not

Each line below is tagged `[CLAIM]` (we assert this, falsifiable by testing) or `[non-CLAIM]` (explicitly NOT claimed; future work).

## Core algorithm correctness

- `[CLAIM]` Sinkhorn-OT implementation in `polyalign.alignment.sinkhorn` matches POT `ot.sinkhorn` reference within `rtol=1e-4` for known small problems.
- `[CLAIM]` Matryoshka prefix-nested alignment recovers prefix-monotonic recon-error trend on synthetic seeds (Bussmann 2025 property).
- `[CLAIM]` OTCP split conformal in `polyalign.conformal.otcp` achieves marginal coverage >= 1 - alpha on synthetic exchangeable data over 100 bootstrap trials, 95% of the time (`tests/test_otcp.py`).
- `[CLAIM]` PAVA monotone isotonic regression (vendored from foldconsensus, Apache-2.0) is a strict copy and inherits foldconsensus's PAVA tests.
- `[CLAIM]` Alignment polytope vertex extraction is deterministic under fixed seed (verified in `tests/test_polytope.py`).
- `[CLAIM]` Cycle-consistency property holds for the polytope-vertex output on synthetic 3-model inputs.

## Cross-architecture

- `[CLAIM]` Sinkhorn-OT cost matrix builder supports `cosine` (default), `l2`, and a stub for `poincare` (v0.2 placeholder). The transformer-vs-SSM compatibility is tested on synthetic high-dimensional inputs.
- `[non-CLAIM]` polyalign v0.1.0a1 does NOT claim real-world SSM-vs-Transformer concept alignment fidelity. The cross-architecture validity at scale will be evaluated in v0.1.1.

## OTCP coverage mode

- `[CLAIM]` marginal coverage is the default in `otcp_calibrate(...)` and is verified on synthetic exchangeable inputs.
- `[non-CLAIM]` per-architecture-pair conditional coverage is implemented but its empirical validity at scale is deferred to v0.1.1.

## Live model integration

- `[non-CLAIM]` Live SAE extraction from Llama-3, Gemma-3, Mamba-2 is NOT included in v0.1.0a1. The `polyalign[llama3]` / `polyalign[mamba]` / `polyalign[saelens]` optional extras install the dependencies but in v0.1.0a1 polyalign only consumes pre-extracted decoder/encoder arrays via `SAEBundle`.

## Ground truth

- `[non-CLAIM]` v0.1.0a1 does NOT claim performance against real hand-labeled cross-model concept pairs. All metrics in `results/v0.1.0a1_ablation.json` are computed against synthetic `[DEMO]`-prefixed pairs from `datasets/ground_truth_v0.1.0a1.jsonl`.
- `[CLAIM]` Every synthetic feature pair in `datasets/ground_truth_v0.1.0a1.jsonl` carries a `[DEMO]` prefix (CI grep enforces).

## Honest-marketing

- `[CLAIM]` README, NOTICE, CHANGELOG, and docs do NOT contain any of the banned phrases listed in `.github/banned-phrases.txt`. CI grep in `.github/workflows/audit.yml` enforces this.
- `[CLAIM]` README contains explicit differentiation against gavagai / AlignSAE / Anthropic crosscoder (CI grep enforces presence of these three terms).

## Deferred to v0.2+

- `[non-CLAIM]` Poincare ball / hyperbolic cost matrix is a `poincare` stub in `polyalign.alignment.sinkhorn`; the actual hyperbolic implementation is deferred to v0.2.
- `[non-CLAIM]` OKLab perceptually-uniform alignment visualization is deferred to v0.2.
- `[non-CLAIM]` Muon (Newton-Schulz orthogonalization) optimizer integration is deferred to v0.2+.

## License

- `[CLAIM]` polyalign is released under Apache License 2.0 (see LICENSE).
- `[CLAIM]` All vendored / referenced upstream projects are listed in NOTICE with their license.
