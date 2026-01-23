# Usage

## Data & configs
- Data is DVC-tracked (GCS remote). Pull: `dvc pull`. Switch remote with `dvc remote default gcs|local`.
- Hydra configs under `configs/` (e.g., `configs/train/train.yaml`). Override via CLI flags: `python -m src.cli train train.batch_size=32 data.train_path=...`.

## Training
- Entry: `python -m src.cli train [overrides]`.
- Profiling: `python -m src.cli train train.profile_steps=10 train.profile_only=true`.
- W&B logging enabled by default; set `WANDB_API_KEY` and optional project/entity/run_name/tags.

## Inference API (local)
- Run FastAPI: `uvicorn src.api:app --host 0.0.0.0 --port 8000`
- Request example: upload `.npz` with `frames` shaped `[T,H,W,3]` (uint8) to `/predict`.
- Env vars: `MODEL_CHECKPOINT` (optional GCS path), `MODEL_NAME`, `K_FRAMES`, `THRESHOLD`.

## Model speedups
- Quantize ONNX → INT8:
  ```bash
  python scripts/quantize_onnx.py \
    --input artifacts/checkpoints/best.onnx \
    --output artifacts/checkpoints/best-int8.onnx
  ```
- Other options mentioned: `torch.compile`, pruning, TensorRT/ORT execution providers (not fully scripted).

## Load testing
- k6: `APP_URL=https://<service>/predict NPZ_PATH=dummy.npz k6 run deploy/loadtest.js`
- Locust: `APP_PATH=/predict NPZ_PATH=dummy.npz locust -f deploy/locustfile.py --host https://<service> --web-host 0.0.0.0 --web-port 8089`
