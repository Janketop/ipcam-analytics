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
        build-essential \
        cmake \
        pkg-config \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libboost-python-dev \
        libx11-dev \
        libjpeg-dev \
        libpng-dev \
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
RUN set -eux; \
    python3 -m pip install --no-cache-dir --default-timeout=${PIP_DEFAULT_TIMEOUT} \
        --force-reinstall -r requirements.txt -c constraints.txt; \
    python3 - <<'PY'
import numpy

expected = "1.26.4"
if numpy.__version__ != expected:
    raise SystemExit(
        f"Pinned NumPy version {expected} is required, got {numpy.__version__}. "
        "Make sure the constraints are applied correctly."
    )
PY
RUN python3 - <<'PY'
import pathlib
import shutil
import nvidia.cudnn

cudnn_lib_dir = pathlib.Path(nvidia.cudnn.__file__).resolve().parent / 'lib'
target_dir = pathlib.Path('/usr/lib/x86_64-linux-gnu')
target_dir.mkdir(parents=True, exist_ok=True)

for so_path in cudnn_lib_dir.glob('libcudnn*.so*'):
    shutil.copy2(so_path, target_dir / so_path.name)
PY

RUN ldconfig

RUN set -eux; \
    python3 -m pip uninstall -y opencv-python || true; \
    python3 -m pip install --no-cache-dir --force-reinstall \
        -c constraints.txt opencv-python-headless==4.10.0.84

COPY . ./backend
EXPOSE 8000
CMD ["python3", "-m", "backend.main"]
