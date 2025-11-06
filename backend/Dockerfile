FROM nvidia/cuda:12.5.0-base-ubuntu22.04

# Устанавливаем CUDA и cuDNN runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-12-5 \
    libcublas-12-5 \
    libcublas-dev-12-5 \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Добавляем в переменные окружения пути к CUDA библиотекам
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

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
    && python3 -m pip install --no-cache-dir --upgrade pip "setuptools<81" wheel \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && apt-get purge -y --auto-remove gnupg software-properties-common \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124

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
# После установки Python-зависимостей переносим cuDNN 9 в системный путь,
# чтобы onnxruntime-gpu мог подхватить libcudnn.so.9 во время запуска.
RUN python3 - <<'PY'
"""Копируем cuDNN (.so) из site-packages в системный путь для onnxruntime."""
import pathlib
import shutil
import sys

package_dirs = [
    pathlib.Path(p)
    for p in sys.path
    if p
    and ("site-packages" in p or "dist-packages" in p)
    and pathlib.Path(p).is_dir()
]

if not package_dirs:
    raise SystemExit(
        "Не найдены каталоги site-packages/dist-packages для поиска libcudnn*.so"
    )

target_dir = pathlib.Path("/usr/lib/x86_64-linux-gnu")
target_dir.mkdir(parents=True, exist_ok=True)

all_libs = []
for base in package_dirs:
    all_libs.extend(sorted(base.rglob("libcudnn*.so*")))

preferred = [p for p in all_libs if ".so.9" in p.name or p.name.endswith(".so")]
selected = preferred or all_libs

if not selected:
    raise SystemExit("Не найдено файлов libcudnn*.so для копирования")

copied = []
for so_path in selected:
    dest = target_dir / so_path.name
    if dest.exists():
        dest.unlink()
    shutil.copy2(so_path, dest)
    dest.chmod(0o755)
    copied.append(dest.name)

print("Скопированы библиотеки cuDNN:", ", ".join(sorted(set(copied))))
PY

RUN ldconfig

RUN set -eux; \
    python3 -m pip uninstall -y opencv-python || true; \
    python3 -m pip install --no-cache-dir --force-reinstall \
        -c constraints.txt opencv-python-headless==4.10.0.84

COPY . ./backend
EXPOSE 8000
CMD ["python3", "-m", "backend.main"]
