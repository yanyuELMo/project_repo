from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import typer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = typer.Typer(help="Project command-line interface wrapper.")


def _run_module(module: str, overrides: List[str]) -> None:
    """Invoke a Python module with optional Hydra overrides."""
    cmd = [sys.executable, "-m", module, *overrides]
    typer.echo(f"[cli] running: {' '.join(cmd)}", err=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


@app.command()
def preprocess(overrides: List[str] = typer.Argument(  # type: ignore[call-arg]
    [],
    help="Hydra overrides, e.g. data.clip_seconds=6 data.force=true",
)) -> None:
    """
    Run data preprocessing (defaults to configs/data/data.yaml).
    Example: python -m src.cli preprocess data.force=true
    """
    _run_module("src.preprocess", overrides)


@app.command()
def train(overrides: List[str] = typer.Argument(  # type: ignore[call-arg]
    [],
    help="Hydra overrides, e.g. train.epochs=10 train.batch_size=16",
)) -> None:
    """
    Run training (defaults to configs/train/train.yaml).
    Example: python -m src.cli train train.lr=5e-5
    """
    _run_module("src.train", overrides)


if __name__ == "__main__":
    app()
