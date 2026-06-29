FROM python:3.11-slim

# opencv-python requires these system libs on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY preprocess.py .

ENTRYPOINT ["python", "preprocess.py"]
