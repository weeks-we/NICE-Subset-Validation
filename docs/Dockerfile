# syntax=docker/dockerfile:1
# Build in a small directory containing this file and runtime.lock.txt only.
# BASE_IMAGE: company's digest-pinned NVIDIA CUDA *devel*, Ubuntu 24.04 image.
# Match the original CUDA/Python/toolchain; do not invent versions for resume.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

# Python, compilation/CUDA-extension prerequisites, Git, Docker CLI, networking,
# scientific-library runtime dependencies and noninteractive plotting support.
# docker.io provides the CLI; no Docker daemon is started inside this image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip \
    build-essential cmake ninja-build pkg-config git git-lfs docker.io ripgrep rsync file \
    ca-certificates curl wget openssh-client bash coreutils findutils \
    patch diffutils jq procps tini unzip zip tar gzip xz-utils \
    libgomp1 libopenblas-dev libssl-dev libffi-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# Set PYTHON_ENV to the original interpreter's venv path if identity requires it.
ARG PYTHON_ENV=/opt/nice-runtime
RUN python3 -c 'import sys; assert sys.version_info[:2] == (3,12), "Use the original Python 3.12 base/toolchain"' \
 && python3 -m venv "${PYTHON_ENV}"
ENV VIRTUAL_ENV=${PYTHON_ENV}
ENV PATH="${PYTHON_ENV}/bin:${PATH}"
WORKDIR /opt/recovery

# Required: complete exact dependencies exported from ORIGINAL validated server.
# Remove only the editable experiment project and swebench/swesmith entries;
# those two official harnesses are installed below with their original commits.
# Preserve kernel VCS URLs/commits in the lock if originally installed that way.
COPY runtime.lock.txt /opt/recovery/runtime.lock.txt
ARG TORCH_VERSION
ARG TORCH_INDEX_URL
ARG SWEBENCH_COMMIT
ARG SWESMITH_COMMIT
ARG TORCH_CUDA_ARCH_LIST=9.0
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}

# Extract ordinary exact version constraints; VCS lines remain in the full lock.
# These build arguments must come from server records, not guessed defaults.
RUN test -s runtime.lock.txt \
 && test -n "${TORCH_VERSION}" && test -n "${TORCH_INDEX_URL}" \
 && [[ "${SWEBENCH_COMMIT}" =~ ^[a-f0-9]{40}$ ]] \
 && [[ "${SWESMITH_COMMIT}" =~ ^[a-f0-9]{40}$ ]] \
 && python -c 'import re; from pathlib import Path; p=Path("runtime.lock.txt"); rows=p.read_text().splitlines(); Path("constraints.txt").write_text("\n".join(s for s in rows if re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s;]+", s))+"\n")' \
 && python -m pip install -c constraints.txt setuptools wheel packaging ninja cmake \
 && python -m pip install -c constraints.txt "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" \
 && python -c 'import torch; assert torch.version.cuda, "CPU PyTorch is not a server runtime"' \
 && python -m pip install --no-build-isolation -r runtime.lock.txt \
 && python -m pip install -c constraints.txt \
    'numpy>=1.26,<3' 'PyYAML>=6,<7' 'matplotlib>=3.8,<4' 'scipy>=1.13,<2' \
    'packaging>=23,<27' 'psutil>=5.9' \
    'torch>=2.6,<3' 'transformers>=5.8,<6' 'peft>=0.18,<1' \
    'accelerate>=1.10,<2' 'datasets>=3,<6' 'huggingface-hub>=0.34,<2' \
    'langchain-core>=1.0,<2' 'docker>=7,<8' 'ghapi<2' \
    'pytest>=8' pytest-cov 'Jinja2>=3.1,<4' \
 && python -m pip install -c constraints.txt \
    "git+https://github.com/SWE-bench/SWE-bench.git@${SWEBENCH_COMMIT}" \
    "swesmith[validate] @ git+https://github.com/SWE-bench/SWE-smith.git@${SWESMITH_COMMIT}" \
 && python -m pip check \
 && python -c 'import torch, transformers, peft, accelerate, datasets, huggingface_hub, numpy, scipy, matplotlib, yaml, docker, swebench, swesmith; from transformers import AutoModelForImageTextToText; assert torch.version.cuda' \
 && python -m pip freeze > /opt/recovery/environment-freeze.txt \
 && dpkg-query -W > /opt/recovery/system-packages.txt

# Original kernel packages (e.g. flash-linear-attention/causal-conv1d) MUST be
# pinned in runtime.lock.txt when the validated model used them. The CUDA devel
# base and torch-first installation above allow their build without a GPU.
# Cross-check actual H200 kernel use/capacity on the server; do not assume that
# package import or a successful build proves original numerical equivalence.
# Mount ORIGINAL experiment code/config/storage/model/data at ORIGINAL paths.
# No updated project, target weights, dataset or task images are copied here.
# Host driver, authorized Docker connection and concrete task images are external.
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "--version"]
