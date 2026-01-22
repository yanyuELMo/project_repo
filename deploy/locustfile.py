"""
Locust load test for the FastAPI /predict endpoint.

Usage:
  APP_PATH=/predict NPZ_PATH=dummy.npz locust -f deploy/locustfile.py --host=https://your-service.run.app
"""

import os
from pathlib import Path

from locust import HttpUser, between, task


APP_PATH = os.environ.get("APP_PATH", "/predict")
NPZ_PATH = os.environ.get("NPZ_PATH", "dummy.npz")


class PredictUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        npz_file = Path(NPZ_PATH)
        if not npz_file.exists():
            raise FileNotFoundError(f"NPZ_PATH missing: {npz_file}")
        self.npz_bytes = npz_file.read_bytes()

    @task
    def predict(self) -> None:
        files = {"file": ("sample.npz", self.npz_bytes, "application/octet-stream")}
        self.client.post(APP_PATH, files=files)
