# Accident Detection with TU-DAT (DTU 02476 MLOps)

This repository contains our course project for **DTU 02476 — Machine Learning Operations (MLOps)**.
The goal is to build an end-to-end MLOps pipeline for a practical computer vision task: **road traffic accident detection from roadside videos**.

Data source:https://github.com/pavana27/TU-DAT
> In this course, we focus on building a reliable ML system (training → evaluation → deployment → monitoring).

## Development

- Formatting: `isort src tests` then `black src tests` (config in `pyproject.toml`).
- Pre-commit: `pre-commit install` once, then `pre-commit run --all-files` to lint/format before committing.
- CI: GitHub Actions workflow `.github/workflows/ci.yml` runs pre-commit (lint/format) and `pytest -q` on push/PR with a Python matrix (3.10, 3.11; Ubuntu runners) and pip cache.
- Data change workflow: `.github/workflows/data-change.yml` triggers on DVC metadata changes (`dvc.yaml`, `dvc.lock`, `*.dvc`, `.dvc/**`) and runs tests.
- Registry change workflow: `.github/workflows/registry-change.yml` triggers on registry updates (`registry/**`, `src/**/registry.*`) and runs tests.
- CLI (Typer wrapper):
  - Preprocess: `python -m src.cli preprocess [hydra overrides]`
  - Train: `python -m src.cli train [hydra overrides]`
  - Report: `python -m src.cli report --mode html|check`
  - Format: `python -m src.cli format`
- Profiling: `python -m src.cli train train.profile_steps=10 train.profile_only=true` (writes `profile.json` under `artifacts/metrics/...`)

## Experiment tracking (Weights & Biases)

- Default: disabled (`wandb.enabled=false` in `configs/train/train.yaml`).
- Enable with overrides, e.g. `python -m src.cli train wandb.enabled=true wandb.project=your_project`.
- Logs clip-level metrics per epoch; can upload best checkpoint/metrics as artifacts when `log_checkpoints`/`log_artifacts` are true.
- Supports `mode=offline` for environments without network access.
- Sweep (W&B): config now under `outputs/sweeps/wandb_clip.yaml`; run with `WANDB_DIR=outputs/wandb wandb sweep outputs/sweeps/wandb_clip.yaml` then `WANDB_DIR=outputs/wandb wandb agent <entity>/<project>/<sweep_id>` (set `WANDB_API_KEY` first).

## Data versioning (DVC)

- Tracked with DVC: `data/raw` and `data/processed/clips`.
- Default remote: GCS bucket `gs://mlops02476-accident-data-eu-7f3a` (auth via ADC or `GOOGLE_APPLICATION_CREDENTIALS=<service-account.json>`).
- Local fallback remote: `/home/vscode/dvc_remote` (remote name `local`).
- Pull data: `dvc pull` (uses default GCS remote); push: `dvc push`.
- To switch remotes: `dvc remote default gcs` or `dvc remote default local`; list with `dvc remote list`.

## Docker image build (Cloud Build)

- Build config: `cloudbuild.yaml` builds `.devcontainer/Dockerfile`.
- Targets: pushes to Artifact Registry `${_REGION}-docker.pkg.dev/$PROJECT_ID/mlops02476-accident-images-eu-7f3a/app` with tags `$COMMIT_SHA` and `latest` (default `_REGION=europe-west1`).
- Usage: create a Cloud Build trigger on push/tag that points to `cloudbuild.yaml`; manual run: `gcloud builds submit --config cloudbuild.yaml .`.
