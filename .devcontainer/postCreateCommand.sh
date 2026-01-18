#!/usr/bin/env bash
set -euo pipefail

TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
CONDA_BIN="/opt/conda/bin/conda"
ENV_NAME="${CONDA_DEFAULT_ENV:-mlops}"
CONDA_RUN="${CONDA_BIN} run -n ${ENV_NAME}"

echo "[postCreate] python=$(${CONDA_RUN} which python)"
${CONDA_RUN} python -m pip install --no-cache-dir torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"

${CONDA_RUN} python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("torch_cuda=", torch.version.cuda)
print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY
