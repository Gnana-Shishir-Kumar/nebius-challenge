# EndoSeg eval + ONNX export job (Nebius Job N3) — CPU is sufficient.
FROM python:3.11-slim

WORKDIR /app
COPY jobs/export/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY jobs/export/export_onnx.py ./jobs/export/export_onnx.py

ENV CHECKPOINT=/checkpoints/unet_best.pt \
    OUT_DIR=/checkpoints \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "jobs/export/export_onnx.py"]
