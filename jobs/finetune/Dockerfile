# EndoSeg fine-tuning job (Nebius Job N2) — GPU.
# Base image ships CUDA + cuDNN + PyTorch; pin a concrete tag in CI.
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY jobs/finetune/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bring in the shared model package and the training entrypoint.
COPY models/ ./models/
COPY jobs/finetune/train_unet.py ./jobs/finetune/train_unet.py

ENV DATA_DIR=/data/processed \
    OUT_DIR=/checkpoints \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "jobs/finetune/train_unet.py"]
