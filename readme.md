# Accident Detection with TU-DAT (DTU 02476 MLOps)

This repository contains our course project for **DTU 02476 — Machine Learning Operations (MLOps)**.
The goal is to build an end-to-end MLOps pipeline for a practical computer vision task: **road traffic accident detection from roadside videos**.

Data source:https://github.com/pavana27/TU-DAT
> In this course, we focus on building a reliable ML system (training → evaluation → deployment → monitoring).

## Development

- Formatting: `isort src tests` then `black src tests` (config in `pyproject.toml`).
- CLI (Typer wrapper):
  - Preprocess: `python -m src.cli preprocess [hydra overrides]`
  - Train: `python -m src.cli train [hydra overrides]`
  - Report: `python -m src.cli report --mode html|check`
  - Format: `python -m src.cli format`

## Experiment tracking (Weights & Biases)

- Default: disabled (`wandb.enabled=false` in `configs/train/train.yaml`).
- Enable with overrides, e.g. `python -m src.cli train wandb.enabled=true wandb.project=your_project`.
- Logs clip-level metrics per epoch; can upload best checkpoint/metrics as artifacts when `log_checkpoints`/`log_artifacts` are true.
- Supports `mode=offline` for environments without network access.

## Data versioning (DVC)

- Tracked with DVC: `data/raw` and `data/processed/clips`.
- Pull data: `dvc pull`
- Push data: `dvc push`
- Default remote is local at `/home/vscode/dvc_remote` (adjust with `dvc remote add`/`dvc remote modify` for S3/GCS).
