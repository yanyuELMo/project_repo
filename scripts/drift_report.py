"""
Run an Evidently data drift report between a reference dataset and a current dataset.

Usage:
  python scripts/drift_report.py \
    --reference baseline.csv \
    --current current.csv \
    --out-html artifacts/evidently_drift_report.html \
    --out-json artifacts/evidently_drift_report.json

Both CSVs must have the same column names. Add/rename/drop columns before running if needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.legacy.report import Report


def run_report(reference: Path, current: Path, out_html: Path, out_json: Path) -> None:
    ref_df = pd.read_csv(reference)
    cur_df = pd.read_csv(current)

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(out_html.as_posix())

    def _to_builtin(obj):
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    serializable = json.loads(json.dumps(report.as_dict(), default=_to_builtin))
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    # Print a quick summary
    try:
        data_drift = report.as_dict()["metrics"][0]["result"]["data"]
        print(f"Data drift share: {data_drift.get('drift_share')}, drift detected: {data_drift.get('dataset_drift')}")
    except Exception:
        pass

    print(f"Saved HTML report to {out_html}")
    print(f"Saved JSON summary to {out_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an Evidently data drift report.")
    parser.add_argument("--reference", type=Path, required=True, help="Path to reference CSV (training/baseline).")
    parser.add_argument("--current", type=Path, required=True, help="Path to current CSV (production sample).")
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path("artifacts/evidently_drift_report.html"),
        help="Output HTML path for the report.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/evidently_drift_report.json"),
        help="Output JSON summary path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_report(args.reference, args.current, args.out_html, args.out_json)
