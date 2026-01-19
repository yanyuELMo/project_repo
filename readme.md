# Accident Detection with TU-DAT (DTU 02476 MLOps)

This repository contains our course project for **DTU 02476 — Machine Learning Operations (MLOps)**.
The goal is to build an end-to-end MLOps pipeline for a practical computer vision task: **road traffic accident detection from roadside videos**.

Data source:https://github.com/pavana27/TU-DAT
> In this course, we focus on building a reliable ML system (training → evaluation → deployment → monitoring).

## Development

- Formatting: `isort src tests` then `black src tests` (config in `pyproject.toml`).

## Data versioning (DVC)

- Tracked with DVC: `data/raw` and `data/processed/clips`.
- Pull data: `dvc pull`
- Push data: `dvc push`
- Default remote is local at `/home/vscode/dvc_remote` (adjust with `dvc remote add`/`dvc remote modify` for S3/GCS).
