import sys
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

import src.cli as cli

runner = CliRunner()


def _stub_run(calls: list[tuple[list[str], Path]], monkeypatch: Any) -> None:
    def _fake(cmd, check, cwd):
        calls.append((cmd, Path(cwd)))
        assert check is True

    monkeypatch.setattr(cli.subprocess, "run", _fake)


def test_preprocess_command_runs_module(monkeypatch):
    calls: list[tuple[list[str], Path]] = []
    _stub_run(calls, monkeypatch)

    result = runner.invoke(cli.app, ["preprocess", "foo=1", "bar=2"])

    assert result.exit_code == 0
    assert len(calls) == 1
    cmd, cwd = calls[0]
    assert cmd == [sys.executable, "-m", "src.preprocess", "foo=1", "bar=2"]
    assert cwd == cli.PROJECT_ROOT


def test_train_command_handles_multirun(monkeypatch):
    calls: list[tuple[list[str], Path]] = []
    _stub_run(calls, monkeypatch)

    result = runner.invoke(cli.app, ["train", "m", "train.lr=1e-4"])

    assert result.exit_code == 0
    cmd, _ = calls[0]
    assert cmd == [
        sys.executable,
        "-m",
        "src.train",
        "-m",
        "train.lr=1e-4",
    ]


def test_report_rejects_invalid_mode():
    result = runner.invoke(cli.app, ["report", "--mode", "bad"])
    assert result.exit_code != 0
    assert "mode must be" in result.output


def test_format_invokes_formatters(monkeypatch):
    calls: list[tuple[list[str], Path]] = []
    _stub_run(calls, monkeypatch)

    result = runner.invoke(cli.app, ["format"])

    assert result.exit_code == 0
    assert calls[0][0][:3] == [sys.executable, "-m", "isort"]
    assert calls[1][0][:3] == [sys.executable, "-m", "black"]
