from __future__ import annotations

import io
import logging
import os
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

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
_MODEL = build_model(_MODEL_NAME, pretrained=False)
_load_checkpoint(_MODEL, _MODEL_CKPT)
_MODEL.eval()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)) -> dict[str, float | int]:
    """
    Accepts an uploaded .npz file with key 'frames' of shape [T, H, W, 3] uint8.
    Returns probability and binary label.
    """
    if not file.filename.endswith(".npz"):
        raise HTTPException(status_code=400, detail="Expected an .npz file")
    try:
        data = np.load(io.BytesIO(file.file.read()), allow_pickle=False)
    except Exception as e:  # pragma: no cover - FastAPI wraps errors
        raise HTTPException(status_code=400, detail=f"Failed to read npz: {e}")
    if "frames" not in data:
        raise HTTPException(status_code=400, detail="Key 'frames' not found in npz")
    frames = data["frames"]
    try:
        x = _preprocess_frames(frames, k_frames=_K_FRAMES)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    with torch.no_grad():
        logits = _MODEL(x)
        prob = torch.sigmoid(logits).item()
    label = int(prob >= _THRESHOLD)
    return {"probability": float(prob), "label": label}


# To run locally: uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
