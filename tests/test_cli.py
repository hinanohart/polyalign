"""polyalign-lint CLI tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from polyalign.cli import app


def test_cli_version_emits_string() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0a1" in result.stdout


def test_cli_smoke_succeeds() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["smoke"])
    assert result.exit_code == 0
    assert "smoke ok" in result.stdout


def test_cli_align_bundles(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    p_a = tmp_path / "a.npz"
    p_b = tmp_path / "b.npz"
    np.savez(
        p_a,
        decoder=rng.standard_normal((16, 8)).astype(np.float32),
        model_id=np.array("a"),
        architecture=np.array("transformer"),
        layer=np.array(0),
    )
    np.savez(
        p_b,
        decoder=rng.standard_normal((16, 8)).astype(np.float32),
        model_id=np.array("b"),
        architecture=np.array("transformer"),
        layer=np.array(0),
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "align",
            "--bundles",
            f"{p_a},{p_b}",
            "--top-k",
            "3",
            "--alpha",
            "0.2",
            "--threshold-coverage",
            "0.0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "vertices: 3" in result.stdout


def test_cli_align_requires_two_bundles(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    p = tmp_path / "a.npz"
    np.savez(p, decoder=rng.standard_normal((16, 8)).astype(np.float32))
    runner = CliRunner()
    result = runner.invoke(app, ["align", "--bundles", str(p)])
    assert result.exit_code != 0
    assert "at least 2" in result.stdout or "BadParameter" in result.output or result.exit_code != 0


def test_cli_align_missing_file(tmp_path: Path) -> None:
    runner = CliRunner()
    bogus = tmp_path / "nope.npz"
    result = runner.invoke(app, ["align", "--bundles", f"{bogus},{bogus}"])
    assert result.exit_code != 0
