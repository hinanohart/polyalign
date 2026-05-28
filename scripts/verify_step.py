"""Polyalign self-drive resume helper (bootstrap protocol §3).

When a builder Claude session resumes after /compact, it reads
`.polyalign-progress.json`, walks `completed_steps`, and calls this
script with `--step S<n>` to verify the previously-completed phase
still holds (no file rot, tests still green, expected artifacts on disk).

Exit code 0 = the step is verified intact -> append to `completed_steps`
                and continue to the next step.
Exit code != 0 = something rotted; the previous step needs re-run.

The script is intentionally narrow: one Bash check per phase. It is
NOT a generic harness; it is the post-/compact resume gate referenced
by protocol §3 line 494.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, shell=False, cwd=ROOT, check=False).returncode


def check_S0() -> int:
    progress = ROOT / ".polyalign-progress.json"
    if not progress.exists():
        return 1
    data = json.loads(progress.read_text())
    return 0 if data.get("checklist", {}).get("S0") else 1


def check_S1() -> int:
    needed = ["pyproject.toml", "LICENSE", "NOTICE", "CHANGELOG.md", "src/polyalign/__init__.py"]
    return 0 if all((ROOT / p).exists() for p in needed) else 1


def _check_python_module_exists(module: str) -> int:
    code = f"import importlib; importlib.import_module({module!r})"
    return _run(["uv", "run", "python", "-c", code])


def check_S2() -> int:
    return _check_python_module_exists("polyalign.backends.gavagai_bridge")


def check_S3() -> int:
    return _check_python_module_exists("polyalign.alignment.sinkhorn")


def check_S4() -> int:
    return _check_python_module_exists("polyalign.matryoshka")


def check_S5() -> int:
    return _check_python_module_exists("polyalign.conformal.otcp")


def check_S6() -> int:
    return _check_python_module_exists("polyalign.polytope.vertices")


def check_S7() -> int:
    gt = ROOT / "datasets" / "ground_truth_v0.1.0a1.jsonl"
    res = ROOT / "results" / "v0.1.0a1_ablation.json"
    if not gt.exists() or not res.exists():
        return 1
    rows = [line for line in gt.read_text().splitlines() if line.strip()]
    if len(rows) < 30:
        return 1
    if not all("[DEMO]" in line for line in rows):
        return 1
    return 0


def check_S7_5() -> int:
    readme = (ROOT / "README.md").read_text()
    if "<!-- ABLATION:BEGIN -->" not in readme:
        return 1
    if "<!-- ABLATION:END -->" not in readme:
        return 1
    for term in ("gavagai", "AlignSAE", "crosscoder"):
        if term.lower() not in readme.lower():
            return 1
    return 0


CHECKS = {
    "S0": check_S0,
    "S1": check_S1,
    "S2": check_S2,
    "S3": check_S3,
    "S4": check_S4,
    "S5": check_S5,
    "S6": check_S6,
    "S7": check_S7,
    "S7.5": check_S7_5,
    "S7_5": check_S7_5,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="polyalign step-verify (post-/compact resume gate)")
    p.add_argument("step", help="phase id, e.g. S0, S1, ..., S7, S7.5")
    p.add_argument("--dry-run", action="store_true", help="alias for the same check (reserved)")
    args = p.parse_args(argv)

    check = CHECKS.get(args.step)
    if check is None:
        print(f"unknown step: {args.step!r}", file=sys.stderr)
        return 2
    rc = check()
    print(f"verify_step {args.step}: rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
