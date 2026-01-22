## ONNX + BentoML service

- Export ONNX:
  ```bash
  python deploy/scripts/export_onnx.py \
    --checkpoint artifacts/checkpoints/outputs/2026-01-19/21-59-25/best.pt \
    --output artifacts/checkpoints/best.onnx \
    --k-frames 8 --height 224 --width 224
  ```
- Serve locally:
  ```bash
  bentoml serve deploy/bento/service:svc --reload
  ```
- Build bento & containerize:
  ```bash
  bentoml build deploy/bento/bentofile.yaml
  bentoml containerize accident-onnx-service:latest
  ```
- Deploy container (example Cloud Run after pushing image to registry):
  ```bash
  gcloud run deploy accident-onnx \
    --image=europe-west10-docker.pkg.dev/PROJECT/REPO/accident-onnx-service:latest \
    --port=3000 --allow-unauthenticated \
    --set-env-vars=MODEL_PATH=/bento/artifacts/checkpoints/best.onnx,K_FRAMES=8,THRESHOLD=0.5
  ```
