"""
Monitoring API: generate a drift report using Evidently.

Environment:
- BASELINE_CSV: path to reference CSV (default: artifacts/baseline.csv)
- LOG_BUCKET: GCS bucket containing request logs under requests/YYYY/MM/DD/*.json
- CURRENT_LIMIT: number of latest logs to fetch (default: 100)

Run locally:
  uvicorn deploy.monitoring.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from google.cloud import storage

from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.legacy.report import Report

app = FastAPI(title="Drift Monitoring", version="0.1.0")

BASELINE_CSV = os.getenv("BASELINE_CSV", "artifacts/baseline.csv")
LOG_BUCKET = os.getenv("LOG_BUCKET")
DEFAULT_LIMIT = int(os.getenv("CURRENT_LIMIT", "100"))


def load_baseline() -> pd.DataFrame:
    try:
        return pd.read_csv(BASELINE_CSV)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load baseline: {exc}")


def fetch_latest_logs(limit: int) -> List[dict]:
    if not LOG_BUCKET:
        raise HTTPException(status_code=500, detail="LOG_BUCKET not set")
    client = storage.Client()
    bucket = client.bucket(LOG_BUCKET)
    blobs = list(bucket.list_blobs(prefix="requests/"))
    # sort by updated timestamp descending
    blobs.sort(key=lambda b: b.updated or datetime.min, reverse=True)
    rows: List[dict] = []
    for blob in blobs[:limit]:
        try:
            data = json.loads(blob.download_as_text())
            rows.append(data)
        except Exception:
            continue
    if not rows:
        raise HTTPException(status_code=404, detail="No logs available for current data")
    return rows


def current_df_from_logs(log_rows: List[dict], ref: pd.DataFrame) -> pd.DataFrame:
    # Expect logs from src/api.py with keys frames_mean/frames_std/frames_shape, etc.
    records = []
    for row in log_rows:
        rec = {
            "mean": row.get("frames_mean"),
            "std": row.get("frames_std"),
            "min": 0.0,
            "max": 255.0,
            "brightness": row.get("frames_mean"),
            "contrast": row.get("frames_std"),
            "height": 224,
            "width": 224,
            "frames": 0,
        }
        shape = row.get("frames_shape")
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            rec["frames"] = shape[0]
            rec["height"] = shape[1]
            rec["width"] = shape[2]
        records.append(rec)
    df = pd.DataFrame(records)
    # Align to reference columns, fill missing with reference means to avoid empty columns
    for col in ref.columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[ref.columns]
    df = df.fillna(ref.mean(numeric_only=True))
    return df


def run_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    ref = reference.copy()
    cur = current.copy()
    # Drop non-numeric/object columns (e.g., file paths) and align columns
    for col in ["file"]:
        ref = ref.drop(columns=[col], errors="ignore")
        cur = cur.drop(columns=[col], errors="ignore")
    common_cols = [c for c in ref.columns if c in cur.columns]
    ref = ref[common_cols].select_dtypes(include=[np.number])
    cur = cur[common_cols].select_dtypes(include=[np.number])

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref, current_data=cur)
    return report


def sanitize_dict(payload):
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, (tuple, set)):
            return [sanitize(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return [sanitize(v) for v in obj.tolist()]
        if obj is pd.NA:
            return None
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, (float, np.floating)):
            return obj if np.isfinite(obj) else None
        if hasattr(obj, "item"):
            val = obj.item()
            return val if not (isinstance(val, float) and not np.isfinite(val)) else None
        return obj

    cleaned = sanitize(payload)
    try:
        return json.loads(json.dumps(cleaned, allow_nan=False))
    except ValueError:
        # fallback: permit NaN in dumps then replace special values with None by parsing via json.loads
        return json.loads(json.dumps(cleaned, allow_nan=True).replace("NaN", "null").replace("Infinity", "null").replace("-Infinity", "null"))


@app.get("/drift", response_class=HTMLResponse)
def drift_report(limit: int = Query(DEFAULT_LIMIT, ge=1, le=1000)) -> HTMLResponse:
    ref = load_baseline()
    logs = fetch_latest_logs(limit=limit)
    cur = current_df_from_logs(logs, ref=ref)
    report = run_report(ref, cur)
    html = report.as_html()
    return HTMLResponse(content=html, media_type="text/html")


@app.get("/drift/json", response_class=JSONResponse)
def drift_report_json(limit: int = Query(DEFAULT_LIMIT, ge=1, le=1000)) -> JSONResponse:
    ref = load_baseline()
    logs = fetch_latest_logs(limit=limit)
    cur = current_df_from_logs(logs, ref=ref)
    report = run_report(ref, cur)
    return JSONResponse(content=sanitize_dict(report.as_dict()))
