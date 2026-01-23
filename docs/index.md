# Accident Detection MLOps

End-to-end MLOps pipeline for road traffic accident detection (videos → model → API → monitoring).

## What’s inside
- Data versioned with DVC (GCS remote); manifests under `data/`.
- Configured training with Hydra + Typer CLI; profiling hooks; logging to Weights & Biases.
- FastAPI inference service (Prometheus metrics); Streamlit frontend.
- CI on GitHub Actions (lint/format/tests/coverage); data/registry triggers.
- Containers built via Cloud Build and published to Container/Artifact Registry; deployed on Cloud Run.
- Monitoring via Managed Prometheus; drift detection API with Evidently; load testing with k6/Locust.

## Quick start (local)
- Install deps: `pip install -r .devcontainer/requirements.txt`
- Pull data (DVC, GCS default): `dvc pull`
- CLI (Typer/Hydra):
  - Preprocess/train/report: `python -m src.cli preprocess|train|report [overrides]`
  - Profiling example: `python -m src.cli train train.profile_steps=10 train.profile_only=true`
- Tests: `pytest -q`
- API (FastAPI): `uvicorn src.api:app --host 0.0.0.0 --port 8000`
- Pre-commit: `pre-commit run --all-files`
- W&B: set `WANDB_API_KEY` (optional `wandb.project/entity/run_name/tags`); disable with `wandb.enabled=false`.

## Docs map
- [Pipeline](pipeline.md): diagram + flow.
- [Usage](usage.md): data, training, API, quantization, load testing.
- [Deployment & Ops](deployment.md): CI/CD, Cloud Build/Run, monitoring, drift, frontend.
