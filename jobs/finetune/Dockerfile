# Build from the repo root:  docker build -f jobs/finetune/Dockerfile .
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY jobs/finetune/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY jobs/finetune/train_unet.py .
COPY models/ ./models/

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "train_unet.py"]
