FROM python:3.11-slim

ENV PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglvnd0 \
        libgl1 \
        libgl1-mesa-dri \
        libgl1-mesa-glx \
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

COPY requirements.txt /app/
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1+cpu \
        torchvision==0.18.1+cpu \
    && pip install --no-cache-dir --default-timeout=${PIP_DEFAULT_TIMEOUT} -r requirements.txt \
    && pip uninstall -y opencv-python || true \
    && pip install --no-cache-dir --force-reinstall opencv-python-headless==4.10.0.84

COPY app /app/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
