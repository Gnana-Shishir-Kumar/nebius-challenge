"""Token-hiding proxy (Nebius N7) — injects bearer auth, forwards to /predict."""

from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

ENDPOINT_URL = os.getenv("ENDPOINT_URL", "")        # e.g. https://your-endpoint.nebius.cloud
NEBIUS_TOKEN = os.getenv("NEBIUS_TOKEN", "")         # secret — never shipped to the client
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
TIMEOUT_S = float(os.getenv("PROXY_TIMEOUT_S", "30"))

app = FastAPI(title="EndoSeg Proxy", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/infer")
async def infer(request: Request) -> Response:
    body = await request.body()
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        resp = await client.post(
            ENDPOINT_URL.rstrip("/") + "/predict",
            content=body,
            headers={
                "Authorization": f"Bearer {NEBIUS_TOKEN}",
                "Content-Type": "application/json",
            },
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
