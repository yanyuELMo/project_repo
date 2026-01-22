"""
Export the trained PyTorch model to ONNX for high-performance serving.

Example:
  python deploy/scripts/export_onnx.py \\
    --checkpoint artifacts/checkpoints/outputs/2026-01-19/21-59-25/best.pt \\
    --output artifacts/checkpoints/best.onnx \\
    --k-frames 8 --height 224 --width 224
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.model import build_model

LOGGER = logging.getLogger(__name__)


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path | None) -> None:
    if ckpt_path is None:
        LOGGER.warning("No checkpoint provided; exporting random-initialized weights.")
        return
    if not ckpt_path.exists():
        LOGGER.warning(
            "Checkpoint not found at %s; exporting random weights.", ckpt_path
        )
        return
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = {k.replace("model.", ""): v for k, v in state["state_dict"].items()}
    model.load_state_dict(state)
    LOGGER.info("Loaded checkpoint from %s", ckpt_path)


def export_onnx(
    checkpoint: Path | None,
    output: Path,
    model_name: str,
    k_frames: int,
    height: int,
    width: int,
) -> None:
    model = build_model(model_name, pretrained=False)
    _load_checkpoint(model, checkpoint)
    model.eval()

    dummy = torch.randn(1, k_frames, 3, height, width)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output.as_posix(),
        input_names=["frames"],
        output_names=["logits"],
        opset_version=18,
        dynamic_axes=None,
    )
    LOGGER.info("Exported ONNX model to %s", output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to PyTorch checkpoint (.pt). Optional; if missing exports random weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/checkpoints/best.onnx"),
        help="Output ONNX path",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="temporal_avg_resnet18",
        help="Model name for build_model",
    )
    parser.add_argument("--k-frames", type=int, default=8, help="Temporal frames")
    parser.add_argument("--height", type=int, default=224, help="Frame height")
    parser.add_argument("--width", type=int, default=224, help="Frame width")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_onnx(
        checkpoint=args.checkpoint,
        output=args.output,
        model_name=args.model_name,
        k_frames=args.k_frames,
        height=args.height,
        width=args.width,
    )
