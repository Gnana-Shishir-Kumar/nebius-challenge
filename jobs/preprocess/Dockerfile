# EndoSeg preprocessing job (Nebius Job N1) — CPU only.
FROM python:3.11-slim

# opencv runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY preprocess.py .

# Data dirs are object-storage mounts provided by the Nebius Job spec.
ENV RAW_DIR=/data/raw/mmotu \
    OUT_DIR=/data/processed

ENTRYPOINT ["python", "preprocess.py"]
