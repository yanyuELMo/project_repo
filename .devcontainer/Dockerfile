FROM mcr.microsoft.com/devcontainers/miniconda:1-3

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
    
RUN conda install -n base -c conda-forge mamba -y && conda clean -afy

COPY .devcontainer/environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml && conda clean -afy

SHELL ["/bin/bash", "-lc"]
RUN echo "conda activate mlops" >> /etc/bash.bashrc

WORKDIR /workspaces
