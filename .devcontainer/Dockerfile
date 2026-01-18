FROM mcr.microsoft.com/devcontainers/miniconda:1-3

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ffmpeg \
    libglib2.0-0 libsm6 libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN conda install -n base -c conda-forge mamba -y && conda clean -afy

ARG ENV_NAME=mlops
ARG PYTHON_VERSION=3.11
RUN mamba create -y -n ${ENV_NAME} python=${PYTHON_VERSION} pip && conda clean -afy

ENV CONDA_DEFAULT_ENV=${ENV_NAME}
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:$PATH

COPY .devcontainer/environment.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspaces
