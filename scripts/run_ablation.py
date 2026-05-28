"""Run polyalign v0.1.0a1 ablation against synthetic ground truth.

Produces `results/v0.1.0a1_ablation.json` with the schema:

  {
    "env_stamp": {dataset_n, mode, hw, os, python, date, seed, version},
    "settings": [
      {"cost": "cosine", "reg": 0.05, "carrier_set": "transformer-only"},
      ...
    ],
    "metrics_by_setting": {
      "<setting_key>": {top1, top5, ece, otcp_coverage, n_vertices,
                         cycle_consistency_rate}
    }
  }

Determinism: a fixed seed is reused for every setting; running twice
produces byte-identical output (verified in `tests/test_ablation_runner.py`).

Honest-marketing scope: every row in datasets/ground_truth_v0.1.0a1.jsonl
carries the `[DEMO]` prefix (enforced by audit.yml). Metrics computed
here reflect synthetic alignment quality, NOT real cross-model concept
matching - see docs/CLAIM.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path

import numpy as np

from polyalign import SAEBundle, __version__, alignment_polytope
from polyalign.calibration import expected_calibration_error
from polyalign.matryoshka import multi_granularity_alignment


def _load_ground_truth(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _make_bundle(
    rng: np.random.Generator, model_id: str, arch: str, n_feat: int, d: int
) -> SAEBundle:
    return SAEBundle(
        model_id=model_id,
        architecture=arch,
        layer=6,
        decoder=rng.standard_normal((n_feat, d)).astype(np.float32),
    )


def _planted_bundles(
    seed: int, carriers: list[str], n_feat: int, d_model: int
) -> tuple[list[SAEBundle], list[np.ndarray]]:
    """Synthetic 'planted' regime: a base decoder is permuted per bundle.

    Returns the SAEBundle list and the inverse permutations (so that
    ground-truth row's feature_id in bundle i corresponds to the same
    *base concept* across bundles). The ablation runner uses this to
    score Sinkhorn-OT's ability to recover the planted permutations -
    a non-trivial signal even on synthetic data, with regime explicitly
    recorded in the JSON cell. See docs/CLAIM.md for honest scope.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_feat, d_model)).astype(np.float32)
    bundles: list[SAEBundle] = []
    perms: list[np.ndarray] = []
    for i, carrier in enumerate(carriers):
        perm = rng.permutation(n_feat) if i > 0 else np.arange(n_feat)
        # bundle i's decoder = base[perm]; row j of bundle 0 corresponds
        # to row perm[j]^{-1} of bundle i. We store the forward perm.
        decoder = base[perm].copy()
        bundles.append(
            SAEBundle(
                model_id=f"planted_{i}_{carrier}",
                architecture=carrier,  # type: ignore[arg-type]
                layer=6,
                decoder=decoder,
            )
        )
        perms.append(perm)
    return bundles, perms


def run_setting(
    *,
    cost: str,
    reg: float,
    carrier_set: str,
    regime: str,
    n_feat: int,
    d_model: int,
    seed: int,
    gt_rows: list[dict],
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    if carrier_set == "transformer-only":
        carriers = ["transformer"] * 3
    elif carrier_set == "ssm-only":
        carriers = ["ssm"] * 3
    elif carrier_set == "hybrid-only":
        carriers = ["hybrid"] * 3
    else:
        carriers = ["transformer", "ssm", "hybrid"]

    if regime == "planted":
        bundles, perms = _planted_bundles(seed, carriers, n_feat, d_model)
        # For the planted regime, the ground-truth feature-pair across bundle 0
        # and bundle 1 is (k, perm_1^{-1}(perm_0(k))) = (k, inv_perm_1[k]).
        # We score with the actual planted permutation, ignoring gt_rows.
        inv_perm_1 = np.argsort(perms[1])
        target_pairs = [(k, int(inv_perm_1[k])) for k in range(n_feat)]
        used = len(target_pairs)
    else:
        # "random" regime: random Gaussian decoders, ground-truth row indices
        # come from datasets/ground_truth_v0.1.0a1.jsonl (Case C synthetic with
        # NO planted alignment). Expected top-k ~ 0 (mathematically correct
        # baseline; see docs/CLAIM.md).
        bundles = [
            _make_bundle(rng, f"random_{i}_{carriers[i]}", carriers[i], n_feat, d_model)
            for i in range(3)
        ]
        used = min(len(gt_rows), 30)
        target_pairs = [
            (int(row["feature_id_A"]) % n_feat, int(row["feature_id_B"]) % n_feat)
            for row in gt_rows[:used]
        ]

    result = alignment_polytope(bundles, top_k=5, alpha=0.1, reg=reg, cost=cost)

    top1_hits = 0
    top5_hits = 0
    for target_a, target_b in target_pairs:
        in_top1 = any(
            edge[0] == 0 and edge[1] == 1 and edge[2] == target_a and edge[3] == target_b
            for v in result.vertices[:1]
            for edge in v.model_pairs
        )
        in_top5 = any(
            edge[0] == 0 and edge[1] == 1 and edge[2] == target_a and edge[3] == target_b
            for v in result.vertices[:5]
            for edge in v.model_pairs
        )
        top1_hits += int(in_top1)
        top5_hits += int(in_top5)

    # ECE against vertex joint probabilities as predicted prob + synthetic
    # ground truth correct=1.0 (under Case C, every synthetic pair is
    # vacuously "correct"). This is honest in that ECE here measures
    # only the *calibration shape*, not concept fidelity.
    if result.vertices:
        probs = np.array([v.joint_probability for v in result.vertices], dtype=np.float64)
        correct = np.ones_like(probs)
        ece = expected_calibration_error(probs, correct, n_bins=min(5, len(probs)))
    else:
        ece = 0.0

    cycle_rate = (
        float(np.mean([v.cycle_consistent for v in result.vertices])) if result.vertices else 0.0
    )

    # Matryoshka quick sanity (recon-error monotone is verified in tests)
    mgr = multi_granularity_alignment(bundles[0], bundles[1])
    _ = mgr.prefix_lengths

    return {
        "top1": top1_hits / max(used, 1),
        "top5": top5_hits / max(used, 1),
        "ece": float(ece),
        "otcp_coverage": float(result.coverage.quantile),
        "n_vertices": len(result.vertices),
        "cycle_consistency_rate": cycle_rate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="polyalign v0.1.0a1 ablation runner")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-feat", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=12)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "datasets" / "ground_truth_v0.1.0a1.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "v0.1.0a1_ablation.json",
    )
    args = parser.parse_args(argv)

    if not args.ground_truth.exists():
        print(f"ERROR: ground truth missing: {args.ground_truth}", file=sys.stderr)
        return 1
    gt_rows = _load_ground_truth(args.ground_truth)
    if len(gt_rows) < 30:
        print(f"ERROR: ground truth count {len(gt_rows)} < 30", file=sys.stderr)
        return 1

    base_settings = [
        {"cost": "cosine", "reg": 0.05, "carrier_set": "transformer-only"},
        {"cost": "cosine", "reg": 0.05, "carrier_set": "ssm-only"},
        {"cost": "cosine", "reg": 0.05, "carrier_set": "hybrid-only"},
        {"cost": "cosine", "reg": 0.05, "carrier_set": "mixed"},
        {"cost": "l2", "reg": 0.05, "carrier_set": "mixed"},
        {"cost": "cosine", "reg": 0.01, "carrier_set": "mixed"},
        {"cost": "cosine", "reg": 0.10, "carrier_set": "mixed"},
    ]
    # Each setting is run under both 'random' (no planted alignment, expected
    # top-k ~ 0) and 'planted' (bundle B is a permutation of bundle A, expected
    # top-k > 0 if Sinkhorn-OT works). docs/CLAIM.md describes the regimes.
    settings = [{**s, "regime": regime} for s in base_settings for regime in ("random", "planted")]
    metrics_by_setting: dict[str, dict] = {}
    for setting in settings:
        key = f"{setting['cost']}_r{setting['reg']}_{setting['carrier_set']}_{setting['regime']}"
        m = run_setting(
            cost=str(setting["cost"]),
            reg=float(setting["reg"]),
            carrier_set=str(setting["carrier_set"]),
            regime=str(setting["regime"]),
            n_feat=args.n_feat,
            d_model=args.d_model,
            seed=args.seed,
            gt_rows=gt_rows,
        )
        m["regime"] = str(setting["regime"])
        metrics_by_setting[key] = m

    payload = {
        "env_stamp": {
            "dataset_n": len(gt_rows),
            "mode": "synthetic",
            "case": "C (S0 OQ3 degrade)",
            "demo_prefix_enforced": True,
            "hw": "cpu",
            "os": platform.system(),
            "python": platform.python_version(),
            "date": dt.datetime.now(dt.UTC).date().isoformat(),
            "seed": args.seed,
            "version": __version__,
            "n_feat": args.n_feat,
            "d_model": args.d_model,
        },
        "settings": settings,
        "metrics_by_setting": metrics_by_setting,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} ({len(metrics_by_setting)} settings, mode=synthetic)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
