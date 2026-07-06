FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git git-lfs ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-gpu.txt /app/requirements-gpu.txt
RUN python3.10 -m pip install --upgrade pip setuptools wheel \
    && python3.10 -m pip install -r /app/requirements-gpu.txt

COPY gpu_server /app/gpu_server

EXPOSE 9000
CMD ["python3.10", "-m", "uvicorn", "gpu_server.app.main:app", "--host", "0.0.0.0", "--port", "9000"]
