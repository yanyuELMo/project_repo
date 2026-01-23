# Pipeline

![MLOps pipeline](assets/overview.png)

## Flow overview
1) **Local dev (containerized)**: work in Docker/devcontainer; pull data from GCS via DVC; run Typer/Hydra CLI for preprocess/train; log metrics/artifacts and sweeps to W&B; produce checkpoints/ONNX with optional quantization/pruning; smoke-test the API image locally with Docker.
2) **GitHub/CI**: push to GitHub; GitHub Actions runs lint/format/tests/coverage (pre-commit, ruff, isort, mypy, pytest) and caches deps; data/registry change triggers are wired in CI.
3) **GCP build & deploy**: Cloud Build Trigger → Cloud Build → Container Registry; Cloud Run pulls the image and the model checkpoint from GCS via env vars.
4) **Serving & monitoring**: Cloud Run inference API (FastAPI) exports metrics to Managed Prometheus; logs/inputs land in GCS; Evidently drift detection service (Cloud Run) reads baseline/current from GCS.
5) **Clients**: Streamlit frontend, load testing (k6/Locust), and end users send HTTPS requests to Cloud Run.
