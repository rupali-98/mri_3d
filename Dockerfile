FROM python:3.11.17-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libraries needed by OpenCV / VTK-style rendering and building some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# CPU build of PyTorch by default; see README for the GPU variant
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# Pinned so rebuilds don't silently pull a new multi-GB release; keep in step with requirements.txt
ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1
RUN pip install --upgrade pip && \
    pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX_URL}

COPY requirements.txt .
RUN pip install -r requirements.txt

# --- Claude Code ---
# Kept below the heavy Python layers so changing it doesn't trigger a full reinstall

# Tools Claude Code uses inside the container (downloads, search, paging, process info)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        ripgrep \
        less \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Non-root user: UID 1000 matches the default WSL user, so files created in /workspace stay editable from Windows
ARG USERNAME=dev
ARG USER_UID=1000
RUN useradd -m -u ${USER_UID} -s /bin/bash ${USERNAME} \
    && mkdir -p /home/${USERNAME}/.claude \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME} /workspace

USER ${USERNAME}
ENV PATH=/home/${USERNAME}/.local/bin:${PATH}
RUN curl -fsSL https://claude.ai/install.sh | bash

# All Claude settings and the login live here, so a volume can keep them across rebuilds
ENV CLAUDE_CONFIG_DIR=/home/${USERNAME}/.claude

# Lets `python -m hippo.*` find the package in src/ without installing it
ENV PYTHONPATH=/workspace/src

EXPOSE 8888 5000

# A shell by default; docker-compose.yml starts JupyterLab instead
CMD ["bash"]
