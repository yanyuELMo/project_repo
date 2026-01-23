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

## Inference API (FastAPI)

- App: `src/api.py` exposes `/health` and `/predict` (upload `.npz` with `frames` [T,H,W,3] uint8). Env vars: `MODEL_CHECKPOINT` (optional), `MODEL_NAME`, `K_FRAMES`, `THRESHOLD`.
- Run locally: `uvicorn src.api:app --host 0.0.0.0 --port 8000`.
- Metrics: Prometheus `/metrics` endpoint (Counters for requests/errors; Histogram for latency) for scraping/Monitoring.

## Cloud Run deployment

- Runtime image (built via `deploy/api/cloudbuild.yaml`): `europe-west10-docker.pkg.dev/gen-lang-client-0354690158/mlops02476-accident-images-eu-7f3a/app:latest`.
- Model checkpoint (GCS): `gs://mlops02476-weights-6789/weights/best.pt` (set `MODEL_CHECKPOINT`).
- Deploy example:
  ```
  gcloud run deploy accident-api \
    --project=gen-lang-client-0354690158 \
    --region=europe-west10 \
    --platform=managed \
    --image=europe-west10-docker.pkg.dev/gen-lang-client-0354690158/mlops02476-accident-images-eu-7f3a/app:latest \
    --port=8000 \
    --cpu=1 --memory=1Gi \
    --allow-unauthenticated \
    --set-env-vars=MODEL_CHECKPOINT=gs://mlops02476-weights-6789/weights/best.pt,THRESHOLD=0.5
  ```
- Cloud Monitoring (GMP sidecar):
  - Deployed service: `accident-api-gmp` (URL: `https://accident-api-gmp-809414772908.europe-west10.run.app`), using official sidecar config `deploy/api/cloudrun-gmp-sidecar.yaml` (scrapes `/metrics` on port 8080 and pushes to Managed Prometheus).
  - Generate traffic so metrics appear: `curl -s -X POST -F file=@dummy.npz https://accident-api-gmp-809414772908.europe-west10.run.app/predict` (the repo ships `dummy.npz`; any NPZ with `frames` works).
  - In Metrics Explorer (PromQL): set time range to “Last 1h”, query `sum by(service_name,endpoint,status)(api_requests_total)` or view histograms via `sum by(le)(rate(api_request_latency_seconds_bucket[5m]))`.
  - If you prefer UI picking: Metrics -> Prometheus Target -> Api -> `api_requests_total/counter` (or latency histogram) -> Apply.

## Load testing (k6)

- Script: `deploy/loadtest.js` (POST `/predict` with a `.npz` file). Env vars:
  - `APP_URL`: Cloud Run predict URL (e.g. `https://...run.app/predict`)
  - `NPZ_PATH`: path to a npz with `frames` [T,H,W,3] uint8
  - Optional: `VUS` (default 5), `DURATION` (default `1m`)
- Example:
  ```bash
  python - <<'PY'
  import numpy as np
  frames = np.random.randint(0,255,(8,32,32,3),dtype=np.uint8)
  np.savez("dummy.npz", frames=frames)
  PY
  APP_URL=https://accident-api-809414772908.europe-west10.run.app/predict \
  NPZ_PATH=dummy.npz \
  k6 run deploy/loadtest.js
  ```

## Load testing (Locust)

- Script: `deploy/locustfile.py` (POST `/predict` with a `.npz` file). Env vars:
  - `NPZ_PATH`: path to a npz with `frames` [T,H,W,3] uint8
  - `APP_PATH`: predict path (default `/predict`)
- Install: `pip install locust` (already listed in `.devcontainer/requirements.txt`).
- Run locally and open http://localhost:8089 to start a test:
  ```bash
  APP_PATH=/predict NPZ_PATH=dummy.npz \
  locust -f deploy/locustfile.py \
    --host https://accident-api-809414772908.europe-west10.run.app \
    --web-host 0.0.0.0 --web-port 8089
  ```

## ONNX + BentoML serving

- Export ONNX: `python deploy/scripts/export_onnx.py --checkpoint artifacts/checkpoints/outputs/2026-01-19/21-59-25/best.pt --output artifacts/checkpoints/best.onnx --k-frames 8 --height 224 --width 224`
- Bento service file: `deploy/bento/service.py`; packaging config: `deploy/bento/bentofile.yaml` (includes `best.onnx`).
- Serve locally: `bentoml serve deploy/bento/service:svc --reload`
- Build bento: `bentoml build deploy/bento/bentofile.yaml`
- Build container: `bentoml containerize accident-onnx-service:latest`, then push to Artifact Registry and deploy to Cloud Run.

## Frontend

- App: `deploy/frontend/frontend.py` (Streamlit). Upload an `.npz` with key `frames` [T,H,W,3] uint8, call backend `/predict`, display probability/label.
- Run locally:
  ```bash
  BACKEND_URL=https://accident-api-809414772908.europe-west10.run.app/predict \
  streamlit run deploy/frontend/frontend.py --server.port 8001 --server.address 0.0.0.0
  ```
- Dependencies: `deploy/frontend/requirements.txt`
- Docker build:
  ```bash
  docker build -t frontend:latest -f deploy/frontend/Dockerfile .
  docker run --rm -p 8001:8001 -e BACKEND_URL=... frontend:latest
  ```

## Data drift check

- Build features from clips: `python scripts/build_baseline_features.py --input-dir data/processed/clips --output baseline.csv --limit 500`
- Drift report: `scripts/drift_report.py` (Evidently). Input: reference CSV and current CSV with aligned columns.
- Example:
  ```bash
  python scripts/drift_report.py \
    --reference baseline.csv \
    --current current.csv \
    --out-html artifacts/evidently_drift_report.html \
    --out-json artifacts/evidently_drift_report.json
  ```
- Dependencies: `evidently` (in `.devcontainer/requirements.txt`)

## Drift monitoring API

- Service: `deploy/monitoring/app.py` (FastAPI). Reads baseline CSV and latest logs from `LOG_BUCKET`, runs Evidently (DataDrift + DataQuality), returns HTML/JSON.
- Env: `BASELINE_CSV` (default `artifacts/baseline.csv`), `LOG_BUCKET` (e.g. `mlops02476-api-logs-eu-0f48`), `CURRENT_LIMIT` (default 100).
- Run locally:
  ```bash
  BASELINE_CSV=artifacts/baseline.csv \
  LOG_BUCKET=mlops02476-api-logs-eu-0f48 \
  uvicorn deploy.monitoring.app:app --host 0.0.0.0 --port 8080
  ```
- Endpoints: `/drift` (HTML report), `/drift/json` (JSON summary).
- Docker: `deploy/monitoring/Dockerfile` (port 8080).
- Cloud Run URL: `https://drift-api-809414772908.europe-west10.run.app`; quick check:
  ```bash
  curl -s https://drift-api-809414772908.europe-west10.run.app/drift/json | head
  ```
