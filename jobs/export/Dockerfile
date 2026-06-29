# Build from the repo root:  docker build -f jobs/export/Dockerfile .
FROM python:3.11-slim

WORKDIR /app
COPY jobs/export/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY jobs/export/export_onnx.py .
COPY models/ ./models/

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "export_onnx.py"]
