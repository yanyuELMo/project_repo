from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer

from src.data import preprocess as run_preprocess

ROOT = Path(__file__).resolve().parent
app = typer.Typer(help="Project commands.")


def _run(cmd: List[str]) -> None:
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


@app.command()
def preprocess(
    clip_seconds: float = 4.0,
    stride_seconds: float = 2.0,
    target_fps: int = 10,
    img_size: int = 224,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    include_challenging: bool = True,
    include_rash: bool = True,
    include_beamng: bool = True,
    force: bool = False,
) -> None:
    """Generate processed clips and manifests."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    run_preprocess(
        clip_seconds=clip_seconds,
        stride_seconds=stride_seconds,
        target_fps=target_fps,
        img_size=img_size,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        include_challenging=include_challenging,
        include_rash=include_rash,
        include_beamng=include_beamng,
        force=force,
    )


@app.command()
def train(
    overrides: Optional[List[str]] = typer.Argument(
        None, help="Hydra overrides, e.g. train.lr=3e-4 train.epochs=20"
    )
) -> None:
    """Run training via Hydra with optional config overrides."""
    cmd = [sys.executable, "src/train.py"]
    if overrides:
        cmd.extend(overrides)
    _run(cmd)


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
