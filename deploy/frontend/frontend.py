"""
Minimal Streamlit frontend: upload an `.npz` with key `frames` [T, H, W, 3] uint8,
send to backend `/predict`, and display probability + label.
"""

from __future__ import annotations

import io
import os

import numpy as np
import requests
import streamlit as st

BACKEND_URL_DEFAULT = (
    os.getenv("BACKEND_URL", "").strip() or "http://127.0.0.1:8000/predict"
)


def normalize_url(url: str) -> str:
    url = url.strip()
    if url and not url.startswith("http"):
        url = "https://" + url
    if url and not url.endswith("/predict"):
        url = url.rstrip("/") + "/predict"
    return url


def send_request(backend_url: str, npz_bytes: bytes) -> tuple[bool, str]:
    try:
        files = {
            "file": ("frames.npz", io.BytesIO(npz_bytes), "application/octet-stream")
        }
        resp = requests.post(backend_url, files=files, timeout=30)
    except Exception as exc:  # pragma: no cover - network
        return False, f"Request failed: {exc}"
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text}"
    return True, resp.text


def main() -> None:
    st.set_page_config(page_title="Accident detection demo", layout="centered")
    st.title("Accident detection demo (frontend)")
    st.markdown(
        "Upload an `.npz` with `frames` [T, H, W, 3] uint8; the backend returns probability and label."
    )

    backend_url_input = st.text_input(
        "Backend /predict URL",
        value=BACKEND_URL_DEFAULT,
        placeholder="https://accident-api-...run.app/predict",
        help="Must be the POST /predict endpoint; 405 means you hit the wrong path or method.",
    )
    backend_url = normalize_url(backend_url_input)

    uploaded = st.file_uploader("Upload .npz (with key 'frames')", type=["npz"])
    st.caption("If empty, click 'Use random sample' to generate a dummy clip.")

    sample_bytes: bytes | None = None
    if st.button("Use random sample"):
        frames = np.random.randint(0, 255, (8, 224, 224, 3), dtype=np.uint8)
        buf = io.BytesIO()
        np.savez(buf, frames=frames)
        sample_bytes = buf.getvalue()
        st.success("Generated random sample.")

    if st.button("Send to backend"):
        if not backend_url:
            st.error("Please set backend URL")
            return
        if uploaded is None and sample_bytes is None:
            st.error("Please upload or generate an .npz")
            return
        npz_bytes = (
            uploaded.getvalue() if uploaded is not None else sample_bytes  # type: ignore[arg-type]
        )
        ok, msg = send_request(backend_url, npz_bytes)
        if ok:
            st.success("Prediction succeeded")
            st.code(msg, language="json")
        else:
            st.error(msg)


if __name__ == "__main__":
    main()
