# EndoSeg inference Endpoint (Nebius Endpoint N4).
# Swap to an onnxruntime-gpu base + CUDAExecutionProvider for the heavy model.
FROM python:3.11-slim

WORKDIR /app
COPY endpoint/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY endpoint/serve.py .
COPY endpoint/analysis.py .

# Model is mounted from object storage at s3://endoseg-models/model/unet.onnx
ENV MODEL_PATH=/data/model/unet.onnx \
    IMG_SIZE=256 \
    PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["python", "serve.py"]
