from __future__ import annotations

import io
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.model import build_model
from src.train import IMAGENET_MEAN, IMAGENET_STD, _sample_indices

app = FastAPI(title="Accident detection API", version="0.1.0")
LOGGER = logging.getLogger(__name__)


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Optional[str]) -> None:
    if not ckpt_path:
        return
    path = os.path.expanduser(ckpt_path)
    # Support GCS paths via fsspec/gcsfs
    if path.startswith("gs://"):
        try:
            import fsspec
        except ImportError:
            LOGGER.warning(
                "fsspec not installed; skipping GCS checkpoint load: %s", path
            )
            return
        with fsspec.open(path, "rb") as f:
            state = torch.load(f, map_location="cpu")
    else:
        if not os.path.exists(path):
            LOGGER.warning("Checkpoint not found: %s (skipping load)", path)
            return
        state = torch.load(path, map_location="cpu")
    # allow checkpoint either as full dict or under "state_dict"
    if isinstance(state, dict) and "state_dict" in state:
        state = {k.replace("model.", ""): v for k, v in state["state_dict"].items()}
    model.load_state_dict(state)
    LOGGER.info("Loaded checkpoint from %s", path)


def _preprocess_frames(
    frames: np.ndarray, k_frames: int, seed: int = 0
) -> torch.Tensor:
    """
    frames: np.ndarray [T, H, W, 3] uint8
    returns tensor [1, k_frames, 3, H, W] float32 normalized
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must be shape [T, H, W, 3]")
    t = int(frames.shape[0])
    sel = _sample_indices(t=t, k=k_frames, train=False, rng=np.random.default_rng(seed))
    frames = frames[sel]
    x = torch.from_numpy(frames).float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()  # [k, 3, H, W]
    mean = IMAGENET_MEAN.to(dtype=x.dtype)
    std = IMAGENET_STD.to(dtype=x.dtype)
    x = (x - mean) / std
    return x.unsqueeze(0)


# Global model (CPU) for simplicity
_MODEL_NAME = os.getenv("MODEL_NAME", "temporal_avg_resnet18")
_MODEL_CKPT = os.getenv("MODEL_CHECKPOINT")  # optional path
_K_FRAMES = int(os.getenv("K_FRAMES", "8"))
_THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
_MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
_LOG_BUCKET = os.getenv("LOG_BUCKET")  # optional GCS bucket for request logs
_LOCAL_LOG_DIR = Path("artifacts/api_logs")
_MODEL = build_model(_MODEL_NAME, pretrained=False)
_load_checkpoint(_MODEL, _MODEL_CKPT)
_MODEL.eval()

# Prometheus metrics
REQUESTS_TOTAL = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "status"]
)
ERRORS_TOTAL = Counter("api_errors_total", "Total API errors", ["endpoint"])
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)


def _log_request_async(record: dict[str, object]) -> None:
    """Write request/response summary to GCS if configured, otherwise local."""
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    record = {"timestamp": ts, **record}
    data = json.dumps(record)

    if _LOG_BUCKET:
        try:
            from google.cloud import storage  # type: ignore

            client = storage.Client()
            bucket = client.bucket(_LOG_BUCKET)
            day_prefix = datetime.utcnow().strftime("%Y/%m/%d")
            blob_name = f"requests/{day_prefix}/log_{uuid.uuid4().hex}.json"
            blob = bucket.blob(blob_name)
            blob.upload_from_string(data, content_type="application/json")
            return
        except Exception as exc:
            LOGGER.warning("Failed to write log to bucket %s: %s", _LOG_BUCKET, exc)

    try:
        _LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _LOCAL_LOG_DIR / f"log_{ts}_{uuid.uuid4().hex}.json"
        log_path.write_text(data, encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("Failed to write local log: %s", exc)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
) -> dict[str, float | int]:
    """
    Accepts an uploaded .npz file with key 'frames' of shape [T, H, W, 3] uint8.
    Returns probability and binary label.
    """
    start = time.perf_counter()
    endpoint = "predict"
    status_label = "200"
    try:
        if not file.filename.endswith(".npz"):
            raise HTTPException(status_code=400, detail="Expected an .npz file")
        data = np.load(io.BytesIO(file.file.read()), allow_pickle=False)
        if "frames" not in data:
            raise HTTPException(status_code=400, detail="Key 'frames' not found in npz")
        frames = data["frames"]
        x = _preprocess_frames(frames, k_frames=_K_FRAMES)
        with torch.no_grad():
            logits = _MODEL(x)
            prob = torch.sigmoid(logits).item()
        label = int(prob >= _THRESHOLD)
        resp = {"probability": float(prob), "label": label}

        if background_tasks is not None:
            record = {
                "request_id": uuid.uuid4().hex,
                "frames_shape": list(frames.shape),
                "frames_mean": float(frames.mean()),
                "frames_std": float(frames.std()),
                "prediction": resp,
                "model_name": _MODEL_NAME,
                "model_version": _MODEL_VERSION,
                "latency_ms": int((time.perf_counter() - start) * 1000),
            }
            background_tasks.add_task(_log_request_async, record)

        return resp
    except HTTPException as exc:
        status_label = str(exc.status_code)
        ERRORS_TOTAL.labels(endpoint=endpoint).inc()
        raise
    except Exception as e:
        status_label = "500"
        ERRORS_TOTAL.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        duration = time.perf_counter() - start
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        REQUESTS_TOTAL.labels(endpoint=endpoint, status=status_label).inc()


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# To run locally: uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
