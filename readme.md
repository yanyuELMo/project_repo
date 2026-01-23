# Accident Detection with TU-DAT (DTU 02476 MLOps)

End-to-end MLOps pipeline for road traffic accident detection (videos → model → API → monitoring).
 Built as the final project for **DTU 02476 — Machine Learning Operations (MLOps)**, focusing on applying a range of MLOps/DevOps tools (Hydra, DVC, CI/CD, Cloud Run, Monitoring, load testing, quantization, etc.).
 Data: https://github.com/pavana27/TU-DAT

## Project structure (essentials)
```
README.md             # You are here
.devcontainer/        # Devcontainer image + requirements
.github/workflows/    # CI + data/registry triggers
configs/              # Hydra configs (train/data/experiments)
  ├─ train/train.yaml # batch/k_frames/num_workers/seed...
  ├─ data/*.yaml      # dataset paths/manifests
  └─ experiment/      # experiment presets (if any)
data/                 # DVC-tracked (raw/processed/manifests)
artifacts/            # Checkpoints/metrics/baselines (ONNX exports live here)
  ├─ checkpoints/
  ├─ metrics/
  └─ baseline.csv
src/                  # Code
  ├─ api.py           # FastAPI inference (+ Prometheus metrics)
  ├─ train.py         # Training loop
  ├─ model.py         # TemporalAvgResNet18
  ├─ cli.py           # Typer entrypoints (preprocess/train/report/format)
  └─ preprocess.py    # Data prep helpers
deploy/               # Deployment assets
  ├─ api/             # Cloud Run Dockerfile, GMP sidecar manifests
  ├─ monitoring/      # Drift detection API (Evidently)
  ├─ frontend/        # Streamlit UI
  ├─ bento/           # BentoML service
  ├─ loadtest.js      # k6 load test
  └─ locustfile.py    # Locust load test
scripts/              # Utilities (baseline features, drift report, ONNX quantization)
tests/                # Pytest (API/train)
outputs/              # Hydra/W&B runs
reports/              # Course report/figures (left untouched)
```

## Quick start (dev)
- Install deps: `pip install -r .devcontainer/requirements.txt`
- Pre-commit: `pre-commit install`; run checks: `pre-commit run --all-files`
- CLI (Typer): `python -m src.cli preprocess|train|report|format [hydra overrides]`
- Tests: `pytest -q`

## CI/CD
- Lint/tests: `.github/workflows/ci.yml` (push/PR, Python 3.10/3.11, Ubuntu).
- Data-trigger: `.github/workflows/data-change.yml` (DVC files).
- Registry-trigger: `.github/workflows/registry-change.yml` (registry/**, src/**/registry.*).

## Data (DVC)
- Tracked: `data/raw`, `data/processed/clips`.
- Remote: `gs://mlops02476-accident-data-eu-7f3a` (ADC or `GOOGLE_APPLICATION_CREDENTIALS`).
- Pull/push: `dvc pull` / `dvc push`. Switch: `dvc remote default gcs|local`.

## Training
- Config: `configs/train/train.yaml` (batch_size, k_frames, num_workers, etc.).
- DataLoader already supports multi-worker (`train.num_workers`); increase for faster I/O.
- Profiling: `python -m src.cli train train.profile_steps=10 train.profile_only=true`.
- W&B logging enabled by default (`wandb.enabled=true`); set `WANDB_API_KEY` and optionally `wandb.project/entity/run_name/tags`. To disable: `wandb.enabled=false` or `wandb.mode=disabled`.

## Inference API (FastAPI)
- Code: `src/api.py` — endpoints `/health`, `/predict` (upload `.npz` with `frames` [T,H,W,3] uint8).
- Env: `MODEL_CHECKPOINT` (optional GCS path), `MODEL_NAME`, `K_FRAMES`, `THRESHOLD`.
- Run locally: `uvicorn src.api:app --host 0.0.0.0 --port 8000`.
- Metrics: Prometheus `/metrics` (requests/errors counters, latency histogram).

## Docker & Cloud Build
- Build config: `cloudbuild.yaml` → image `${_REGION}-docker.pkg.dev/$PROJECT_ID/mlops02476-accident-images-eu-7f3a/app:latest`.
- Manual: `gcloud builds submit --config cloudbuild.yaml .`

## Cloud Run (inference)
- Image: `europe-west10-docker.pkg.dev/gen-lang-client-0354690158/mlops02476-accident-images-eu-7f3a/app:latest`
- Example deploy:
  ```
  gcloud run deploy accident-api \
    --project=gen-lang-client-0354690158 --region=europe-west10 --platform=managed \
    --image=europe-west10-docker.pkg.dev/gen-lang-client-0354690158/mlops02476-accident-images-eu-7f3a/app:latest \
    --port=8000 --cpu=1 --memory=1Gi --allow-unauthenticated \
    --set-env-vars=MODEL_CHECKPOINT=gs://mlops02476-weights-6789/weights/best.pt,THRESHOLD=0.5
  ```

## Monitoring (Managed Prometheus)
- Deployed service with sidecar: `accident-api-gmp` (`https://accident-api-gmp-809414772908.europe-west10.run.app`), config `deploy/api/cloudrun-gmp-sidecar.yaml` (scrapes `/metrics` on 8080).
- Generate traffic to see data: `curl -s -X POST -F file=@dummy.npz https://accident-api-gmp-809414772908.europe-west10.run.app/predict`.
- Metrics Explorer (PromQL): `sum by(service_name,endpoint,status)(api_requests_total)`; histogram: `sum by(le)(rate(api_request_latency_seconds_bucket[5m]))`.
- UI picking: Prometheus Target → Api → `api_requests_total/counter` (or latency histogram).

## Model speedups
- Quantize ONNX → INT8:
  ```bash
  python scripts/quantize_onnx.py \
    --input artifacts/checkpoints/best.onnx \
    --output artifacts/checkpoints/best-int8.onnx
  ```
  (merges external weights, drops value_info). Outputs: `best-int8.onnx` (~11MB, INT8), `best-int8-merged.onnx` (~43MB, FP32 merged). Validate accuracy before deploy; serve with ONNX Runtime/TensorRT.
- Other options: `torch.compile`, pruning, TensorRT/ORT execution providers (not scripted here).

## Load testing
- k6: `deploy/loadtest.js`; env `APP_URL`, `NPZ_PATH` (e.g. `dummy.npz`), optional `VUS`, `DURATION`.
  ```bash
  APP_URL=https://accident-api-809414772908.europe-west10.run.app/predict \
  NPZ_PATH=dummy.npz \
  k6 run deploy/loadtest.js
  ```
- Locust: `deploy/locustfile.py`; env `NPZ_PATH`, `APP_PATH` (default `/predict`).
  ```bash
  APP_PATH=/predict NPZ_PATH=dummy.npz \
  locust -f deploy/locustfile.py \
    --host https://accident-api-809414772908.europe-west10.run.app \
    --web-host 0.0.0.0 --web-port 8089
  ```

## Frontend (Streamlit)
- App: `deploy/frontend/frontend.py` — upload `.npz` (`frames` key), call backend `/predict`, show prob/label.
- Run: `BACKEND_URL=https://accident-api-809414772908.europe-west10.run.app/predict streamlit run deploy/frontend/frontend.py --server.port 8001 --server.address 0.0.0.0`
- Docker: `deploy/frontend/Dockerfile` → run with `BACKEND_URL=...`.

## Drift detection
- Build baseline: `python scripts/build_baseline_features.py --input-dir data/processed/clips --output baseline.csv --limit 500`
- Report: `python scripts/drift_report.py --reference baseline.csv --current current.csv --out-html artifacts/evidently_drift_report.html --out-json artifacts/evidently_drift_report.json`
- API: `deploy/monitoring/app.py` (FastAPI) — env `BASELINE_CSV`, `LOG_BUCKET` (e.g. `mlops02476-api-logs-eu-0f48`), `CURRENT_LIMIT`. Run: `uvicorn deploy.monitoring.app:app --host 0.0.0.0 --port 8080`
- Cloud Run drift API: `https://drift-api-809414772908.europe-west10.run.app` (`/drift`, `/drift/json`).

## ONNX + BentoML serving
- Export ONNX: `python deploy/scripts/export_onnx.py --checkpoint artifacts/checkpoints/outputs/2026-01-19/21-59-25/best.pt --output artifacts/checkpoints/best.onnx --k-frames 8 --height 224 --width 224`
- Bento service: `deploy/bento/service.py`; build: `bentoml build deploy/bento/bentofile.yaml`; serve: `bentoml serve deploy/bento/service:svc --reload`; containerize: `bentoml containerize accident-onnx-service:latest`.

## Notes
- Sample NPZ for testing: `dummy.npz` (root).
- Model: `src/model.py` (TemporalAvgResNet18) with Hydra configs in `configs/`.
- Keep content in English; course report is in `reports/` (untouched).
