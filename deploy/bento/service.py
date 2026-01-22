"""
BentoML service for ONNX-based accident detection.

Exposes /predict expecting a NumPy array with key "frames" shaped [T, H, W, 3] uint8.
Sampling, normalization, and ONNX inference are handled inside the service.
"""

from __future__ import annotations

import os
from pathlib import Path

import bentoml
import numpy as np
import onnxruntime as ort
from bentoml.io import NumpyNdarray


def _sample_indices(t: int, k: int) -> np.ndarray:
    if t <= 0:
        raise ValueError("Clip has 0 frames.")
    return np.linspace(0, t - 1, num=k).round().astype(int)


def _preprocess(frames: np.ndarray, k_frames: int) -> np.ndarray:
    """
    frames: [T, H, W, 3] uint8
    returns: [1, k_frames, 3, H, W] float32 normalized
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must be shape [T, H, W, 3]")
    t = int(frames.shape[0])
    sel = _sample_indices(t, k_frames)
    frames = frames[sel]
    x = frames.astype(np.float32) / 255.0  # [k, H, W, 3]
    x = np.transpose(x, (0, 3, 1, 2))  # [k, 3, H, W]
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(
        1, 3, 1, 1
    )
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    x = (x - imagenet_mean) / imagenet_std
    return x[None, ...]  # [1, k, 3, H, W]


K_FRAMES = int(os.getenv("K_FRAMES", "8"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts" / "checkpoints" / "best.onnx"
)
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH.as_posix())

try:
    SESSION = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
except Exception as exc:  # pragma: no cover - startup failure
    raise RuntimeError(f"Failed to load ONNX model at {MODEL_PATH}") from exc

svc = bentoml.Service("accident-onnx-service")


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(frames: np.ndarray) -> np.ndarray:
    """
    Accepts an array shaped [T, H, W, 3] uint8 and returns probability + label.
    """
    x = _preprocess(frames, k_frames=K_FRAMES)
    inputs = {SESSION.get_inputs()[0].name: x}
    logits = SESSION.run(None, inputs)[0].squeeze()
    prob = 1 / (1 + np.exp(-logits))
    label = (prob >= THRESHOLD).astype(np.int64)
    return np.array([prob, label], dtype=np.float32)
