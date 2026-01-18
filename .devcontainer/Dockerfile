FROM mcr.microsoft.com/devcontainers/miniconda:1-3

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ffmpeg \
    libglib2.0-0 libsm6 libxext6 \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN conda install -n base -c conda-forge mamba -y \
  && conda clean -afy \
  && conda config --system --set auto_activate_base false


ARG ENV_NAME=mlops
ARG PYTHON_VERSION=3.11
ENV CONDA_ENV_PATH=/opt/conda/envs/${ENV_NAME}
RUN mamba create -y -n ${ENV_NAME} python=${PYTHON_VERSION} pip && conda clean -afy

ENV CONDA_DEFAULT_ENV=${ENV_NAME}
ENV PATH=${CONDA_ENV_PATH}/bin:/opt/conda/bin:$PATH

RUN echo ". /opt/conda/etc/profile.d/conda.sh && conda activate ${ENV_NAME}" > /etc/profile.d/activate-mlops.sh

COPY .devcontainer/requirements.txt /tmp/requirements.txt
RUN conda run -n ${ENV_NAME} python -m pip install --upgrade pip \
 && conda run -n ${ENV_NAME} python -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspaces
