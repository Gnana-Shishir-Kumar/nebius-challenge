# EndoSeg token-hiding proxy (Nebius feature N7).
FROM python:3.11-slim

WORKDIR /app
COPY proxy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proxy/proxy.py .

# NEBIUS_TOKEN must be supplied at runtime as a secret — never bake it in.
ENV NEBIUS_ENDPOINT_URL=https://your-endpoint.nebius.cloud \
    ALLOWED_ORIGINS=* \
    PORT=8080 \
    PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["python", "proxy.py"]
