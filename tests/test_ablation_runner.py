"""Sanity tests for the v0.1.0a1 ablation runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_ground_truth_jsonl_exists_and_has_30_rows() -> None:
    p = REPO_ROOT / "datasets" / "ground_truth_v0.1.0a1.jsonl"
    assert p.exists(), f"missing: {p}"
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    assert len(rows) >= 30, f"only {len(rows)} rows; S0 OQ3 requires >= 30"


def test_every_ground_truth_row_has_demo_prefix() -> None:
    p = REPO_ROOT / "datasets" / "ground_truth_v0.1.0a1.jsonl"
    for i, line in enumerate(p.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        assert row["concept_label"].startswith("[DEMO]"), (
            f"row {i} concept_label missing [DEMO] prefix: {row['concept_label']}"
        )
        assert row["prompt"].startswith("[DEMO]"), f"row {i} prompt missing [DEMO] prefix"


def test_ablation_results_json_exists() -> None:
    p = REPO_ROOT / "results" / "v0.1.0a1_ablation.json"
    assert p.exists(), "run scripts/run_ablation.py first"
    payload = json.loads(p.read_text())
    assert payload["env_stamp"]["mode"] == "synthetic"
    assert payload["env_stamp"]["demo_prefix_enforced"] is True
    for setting_key, metrics in payload["metrics_by_setting"].items():
        for k in ("top1", "top5", "ece", "otcp_coverage", "n_vertices", "cycle_consistency_rate"):
            assert k in metrics, f"setting {setting_key} missing metric {k}"
            assert metrics[k] is not None, f"setting {setting_key}.{k} is null"


def test_ablation_runner_is_deterministic(tmp_path: Path) -> None:
    """Two runs with identical args produce metric-equal JSON."""
    out1 = tmp_path / "a.json"
    out2 = tmp_path / "b.json"
    for out in (out1, out2):
        rc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_ablation.py"),
                "--seed",
                "42",
                "--out",
                str(out),
            ],
            check=False,
        ).returncode
        assert rc == 0
    p1 = json.loads(out1.read_text())
    p2 = json.loads(out2.read_text())
    assert p1["metrics_by_setting"] == p2["metrics_by_setting"]
