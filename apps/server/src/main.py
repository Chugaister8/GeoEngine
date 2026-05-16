"""
GeoEngine — FastAPI Application Entry Point
Запуск: uvicorn apps.server.src.main:app --reload
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from .config import settings
from .ws.handler import websocket_endpoint
from .api.terrain import router as terrain_router

# ---- Логування ----
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if settings.debug
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger(__name__)

# ----------------------------------------------------------------
# FASTAPI APP
# ----------------------------------------------------------------

app = FastAPI(
    title       = "GeoEngine API",
    description = "3D геопросторовий рушій — REST API + WebSocket",
    version     = "0.1.0",
    docs_url    = "/docs"    if settings.debug else None,
    redoc_url   = "/redoc"   if settings.debug else None,
)

# ---- Middleware ----
app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.cors_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ---- Routers ----
app.include_router(terrain_router)

# ---- WebSocket ----
app.add_api_websocket_route("/ws", websocket_endpoint)


# ---- Lifecycle ----
@app.on_event("startup")
async def startup() -> None:
    settings.dem_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "geoengine.server.start",
        host=settings.host,
        port=settings.port,
        debug=settings.debug,
        dem_cache=str(settings.dem_cache_dir),
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    log.info("geoengine.server.stop")


# ---- Health ----
@app.get("/health")
async def health() -> dict:
    from .ws.handler import manager
    return {
        "status":      "ok",
        "version":     "0.1.0",
        "connections": manager.connection_count,
    }


# ---- Dev entry ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "apps.server.src.main:app",
        host    = settings.host,
        port    = settings.port,
        reload  = settings.debug,
        workers = 1,
    )
