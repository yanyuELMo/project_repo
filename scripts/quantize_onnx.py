"""Quantize an ONNX model (with external data) to INT8 for faster inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize(input_path: Path, output_path: Path) -> None:
    """Run dynamic quantization and write the INT8 model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge external data into a single ONNX and drop existing value_info
    # to avoid shape mismatch during inference in quantization.
    merged_path = output_path.with_name(output_path.stem + "-merged.onnx")
    model = onnx.load_model(input_path, load_external_data=True)
    model.graph.ClearField("value_info")
    onnx.save(model, merged_path, save_as_external_data=False)

    quantize_dynamic(
        model_input=str(merged_path),
        model_output=str(output_path),
        per_channel=False,
        reduce_range=True,
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model written to {output_path}")

    # Keep merged copy for reference; comment out to delete if undesired.
    print(f"Merged model kept at {merged_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/checkpoints/best.onnx"),
        help="Path to the source ONNX model (can reference external data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/checkpoints/best-int8.onnx"),
        help="Path for the quantized ONNX model.",
    )
    args = parser.parse_args()
    quantize(args.input, args.output)
