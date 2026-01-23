# Deployment & Ops

## CI/CD
- GitHub Actions: lint/format/tests/coverage (pre-commit, ruff, isort, mypy, pytest).
- Additional triggers for data and registry changes.
- Cloud Build Trigger kicks off image builds on main.

## Cloud Build & Registry
- Config: `cloudbuild.yaml` builds the FastAPI image.
- Output: image pushed to Container/Artifact Registry.

## Cloud Run (inference)
- Deploy image to Cloud Run. Example:
  ```bash
  gcloud run deploy accident-api \
    --project=<project> --region=<region> --platform=managed \
    --image=<region>-docker.pkg.dev/<project>/mlops02476-accident-images-eu-7f3a/app:latest \
    --port=8000 --cpu=1 --memory=1Gi --allow-unauthenticated \
    --set-env-vars=MODEL_CHECKPOINT=gs://<bucket>/weights/best.pt,THRESHOLD=0.5
  ```
- Cloud Run pulls the container from Registry and the checkpoint from GCS.

## Monitoring
- FastAPI exposes Prometheus metrics (`/metrics` counters + latency histograms).
- Managed Prometheus scrapes via Cloud Run sidecar (see `deploy/api/cloudrun-gmp-sidecar.yaml` in repo).
- Use Metrics Explorer/PromQL for latency, errors, and request volume.

## Drift detection
- Build baseline: `python scripts/build_baseline_features.py --input-dir data/processed/clips --output baseline.csv --limit 500`
- Drift report: `python scripts/drift_report.py --reference baseline.csv --current current.csv --out-html artifacts/evidently_drift_report.html --out-json artifacts/evidently_drift_report.json`
- Drift API: `deploy/monitoring/app.py`; deploy to Cloud Run (`BASELINE_CSV`, `LOG_BUCKET`, `CURRENT_LIMIT` envs).

## Frontend & load
- Streamlit frontend (`deploy/frontend/frontend.py`) calls the Cloud Run API; Dockerfile available under `deploy/frontend/`.
- Load testing via k6 (`deploy/loadtest.js`) and Locust (`deploy/locustfile.py`).
