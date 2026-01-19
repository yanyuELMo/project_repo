from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = typer.Typer(help="Project command-line interface wrapper.")


def _run(cmd: list[str]) -> None:
    """Run a command in the project root and fail fast on errors."""
    typer.echo(f"[cli] running: {' '.join(cmd)}", err=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def _run_module(module: str, overrides: List[str]) -> None:
    """Invoke a Python module with optional Hydra overrides."""
    _run([sys.executable, "-m", module, *overrides])


@app.command()
def preprocess(
    ctx: typer.Context,
    overrides: Optional[List[str]] = typer.Argument(  # type: ignore[call-arg]
        None,
        help="Hydra overrides, e.g. data.clip_seconds=6 data.force=true",
    ),
) -> None:
    """
    Run data preprocessing (defaults to configs/data/data.yaml).
    Example: python -m src.cli preprocess data.force=true
    """
    base = [(o or "").lstrip("-") for o in overrides or []]
    extra = [a.lstrip("-") for a in ctx.args]
    merged = base + extra
    _run_module("src.preprocess", merged)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train(
    ctx: typer.Context,
    overrides: Optional[List[str]] = typer.Argument(  # type: ignore[call-arg]
        None,
        help="Hydra overrides, e.g. train.epochs=10 train.batch_size=16",
    ),
) -> None:
    """
    Run training (defaults to configs/train/train.yaml).
    Example: python -m src.cli train train.lr=5e-5
    """
    base = [(o or "").lstrip("-") for o in overrides or []]
    extra = [a.lstrip("-") for a in ctx.args]
    merged = base + extra
    _run_module("src.train", merged)


@app.command()
def report(mode: str = typer.Option("html", help="One of: html, check")) -> None:
    """Generate or check the report."""
    if mode not in {"html", "check"}:
        raise typer.BadParameter("mode must be 'html' or 'check'")
    _run([sys.executable, "reports/report.py", mode])


@app.command()
def format() -> None:
    """Run isort then black on source and tests."""
    _run([sys.executable, "-m", "isort", "src", "tests"])
    _run([sys.executable, "-m", "black", "src", "tests"])


if __name__ == "__main__":
    app()
