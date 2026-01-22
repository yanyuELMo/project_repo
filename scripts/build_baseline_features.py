"""
Extract simple features from .npz clips to build a baseline/current CSV for drift analysis.

Usage:
  python scripts/build_baseline_features.py \
    --input-dir data/processed/clips \
    --output baseline.csv \
    --limit 500

Each .npz is expected to contain a key 'frames' with shape [T, H, W, 3] uint8.
Features: mean, std, min, max, brightness (mean), contrast (std), height, width, frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def iter_npz_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.npz"):
        if path.is_file():
            yield path


def extract_features(npz_path: Path) -> dict[str, float]:
    data = np.load(npz_path, allow_pickle=False)
    if "frames" not in data:
        raise KeyError(f"'frames' key missing in {npz_path}")
    frames = data["frames"]
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Unexpected frames shape {frames.shape} in {npz_path}")

    mean = float(frames.mean())
    std = float(frames.std())
    return {
        "file": npz_path.as_posix(),
        "mean": mean,
        "std": std,
        "min": float(frames.min()),
        "max": float(frames.max()),
        "brightness": mean,
        "contrast": std,
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "frames": int(frames.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract features from .npz clips for drift analysis."
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Directory containing .npz clips."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional limit on number of files."
    )
    args = parser.parse_args()

    rows = []
    for i, npz_path in enumerate(iter_npz_files(args.input_dir)):
        try:
            rows.append(extract_features(npz_path))
        except Exception as exc:
            print(f"Skip {npz_path}: {exc}")
        if args.limit and len(rows) >= args.limit:
            break

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
