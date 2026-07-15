#!/usr/bin/env bash
set -eu

export PATH="/home/shish/.nebius/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
cd /mnt/d/Projects/nebius-challenge

REGION_ID=eu-north1
REGISTRY_PATH=e00jnhyx0bf601aj7q
IMAGE="cr.${REGION_ID}.nebius.cloud/${REGISTRY_PATH}/endoseg-endpoint:latest"

MODEL_BUCKET="$(nebius storage bucket get-by-name --name endoseg-models --format jsonpath='{.metadata.id}')"
echo "MODEL_BUCKET=${MODEL_BUCKET}"

echo "Pushing ${IMAGE} ..."
docker push "${IMAGE}"

AUTH_TOKEN="$(openssl rand -hex 32)"
echo "AUTH_TOKEN=${AUTH_TOKEN}"
echo "(copy the token above; it will not be written to disk)"

echo "Creating endpoint ..."
nebius ai endpoint create \
  --name endoseg-endpoint \
  --image "${IMAGE}" \
  --container-port 8000 \
  --env MODEL_PATH=/data/model/unet.onnx \
  --env IMG_SIZE=256 \
  --env MODEL_VERSION=unet-v1 \
  --env PORT=8000 \
  --env PYTHONUNBUFFERED=1 \
  --volume "${MODEL_BUCKET}:/data:ro" \
  --platform cpu-d3 \
  --preset 4vcpu-16gb \
  --auth token \
  --token "${AUTH_TOKEN}"

echo "Done. Endpoint details:"
nebius ai endpoint get-by-name --name endoseg-endpoint || nebius ai endpoint list
