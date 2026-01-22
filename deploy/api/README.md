## Cloud Run deployment notes

- Image: `europe-west10-docker.pkg.dev/gen-lang-client-0354690158/mlops02476-accident-images-eu-7f3a/app:latest` (built with `deploy/api/cloudbuild.yaml`).
- Model checkpoint (GCS): `gs://mlops02476-weights-6789/weights/best.pt` (set via `MODEL_CHECKPOINT` env var).
- Default env vars: `PORT=8000`, optional `THRESHOLD`/`K_FRAMES`/`MODEL_NAME`.

Deploy command (example):
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

Health check:
```
curl https://<SERVICE_URL>/health
```
Prediction (upload `.npz` with key `frames` [T,H,W,3] uint8):
```
curl -X POST -F "file=@sample.npz" https://<SERVICE_URL>/predict
```
