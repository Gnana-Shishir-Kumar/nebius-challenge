"""Thin token-hiding proxy (Nebius feature N7).

The browser must never see the Nebius Endpoint token. This tiny FastAPI app
sits between the static site and the Endpoint: it injects the secret
`Authorization` header server-side, forwards the request, and streams the
response back. Deploy it anywhere with the token set as an env secret.

Keep it dumb on purpose — no business logic, just auth injection + forwarding
(plus permissive CORS so the static demo can call it).
"""

from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

NEBIUS_ENDPOINT_URL = os.getenv("NEBIUS_ENDPOINT_URL", "https://your-endpoint.nebius.cloud")
NEBIUS_TOKEN = os.getenv("NEBIUS_TOKEN", "")  # secret — never ship to the client
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
TIMEOUT_S = float(os.getenv("PROXY_TIMEOUT_S", "60"))

app = FastAPI(title="EndoSeg Proxy", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "configured": bool(NEBIUS_TOKEN)}


@app.post("/infer")
async def proxy_infer(request: Request) -> Response:
    """Forward the multipart body to the Endpoint with the secret token added."""
    body = await request.body()
    headers = {
        "Authorization": f"Bearer {NEBIUS_TOKEN}",
        "Content-Type": request.headers.get("content-type", "application/octet-stream"),
    }
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        upstream = await client.post(
            f"{NEBIUS_ENDPOINT_URL.rstrip('/')}/infer",
            content=body,
            headers=headers,
        )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
