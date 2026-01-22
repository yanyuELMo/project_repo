import io

import numpy as np
from fastapi.testclient import TestClient

from src.api import app


def _make_npz() -> bytes:
    frames = np.random.randint(0, 255, (8, 16, 16, 3), dtype=np.uint8)
    buf = io.BytesIO()
    np.savez(buf, frames=frames)
    buf.seek(0)
    return buf.getvalue()


def test_health(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_predict_success(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    client = TestClient(app)
    files = {"file": ("dummy.npz", _make_npz(), "application/octet-stream")}
    res = client.post("/predict", files=files)
    assert res.status_code == 200
    body = res.json()
    assert "probability" in body and "label" in body
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in (0, 1)


def test_predict_missing_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    client = TestClient(app)
    buf = io.BytesIO()
    np.savez(buf, other=np.zeros((1, 1, 1, 3), dtype=np.uint8))
    buf.seek(0)
    files = {"file": ("bad.npz", buf.getvalue(), "application/octet-stream")}
    res = client.post("/predict", files=files)
    assert res.status_code == 400
    assert "frames" in res.json()["detail"]
