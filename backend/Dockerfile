FROM nvidia/cuda:12.5.0-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3.11-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python3 -m ensurepip --upgrade \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && apt-get purge -y --auto-remove gnupg software-properties-common \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglvnd0 \
        libgl1 \
        libgl1-mesa-dri \
        libglu1-mesa \
        libglib2.0-0 \
        libgomp1 \
        libice6 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig \
    && ldconfig -p | grep -q "libGL.so.1"

COPY requirements.txt constraints.txt ./
RUN python3 -m pip install --no-cache-dir --default-timeout=${PIP_DEFAULT_TIMEOUT} -r requirements.txt -c constraints.txt \
    && python3 -m pip uninstall -y opencv-python || true \
    && python3 -m pip install --no-cache-dir --force-reinstall opencv-python-headless==4.10.0.84

COPY . ./backend
EXPOSE 8000
CMD ["python3", "-m", "backend.main"]
