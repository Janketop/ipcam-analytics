FROM nvidia/cuda:12.5.0-base-ubuntu22.04

RUN apt update \
    && apt install -y python3 python3-pip python3-venv \
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
