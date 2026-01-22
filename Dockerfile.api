FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps for numpy/torch/opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libglib2.0-0 libsm6 libxext6 \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY .devcontainer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
 && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy source
COPY src ./src
COPY configs ./configs

ENV PYTHONPATH=/app
ENV PORT=8000

# Default to CPU inference; env vars can override in Cloud Run
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
