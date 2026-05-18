"""
GeoEngine — FastAPI Application
Головний entry point сервера.

Запуск:
    uvicorn apps.server.src.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from .config        import config
from .api.terrain   import router as terrain_router, set_terrain_service
from .api.osm       import router as osm_router
from .services.terrain  import TerrainService
from .services.analysis import AnalysisService
from .ws.handler    import websocket_endpoint, manager
from geoengine.utils.logging import configure_logging

# Ініціалізація логування
configure_logging(
    level=config.log_level,
    json_logs=config.json_logs,
    service="geoengine-server",
)
log: structlog.BoundLogger = structlog.get_logger(__name__)


# ----------------------------------------------------------------
# LIFESPAN
# ----------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifecycle: startup → yield → shutdown.
    Створює та закриває всі сервіси.
    """
    log.info("server.startup", version="0.1.0", debug=config.debug)

    # Ініціалізувати TerrainService
    terrain_svc = await TerrainService.create(
        dem_cache_dir=config.dem_cache_dir,
        max_workers=config.terrain_workers,
        api_keys=config.dem_api_keys,
    )
    analysis_svc = AnalysisService(
        terrain_service=terrain_svc,
        max_workers=config.analysis_workers,
    )

    # Зберегти у app state та DI
    app.state.terrain  = terrain_svc
    app.state.analysis = analysis_svc
    set_terrain_service(terrain_svc)

    log.info("server.ready", host=config.host, port=config.port)
    yield

    # Shutdown
    log.info("server.shutdown")
    await terrain_svc.close()
    await analysis_svc.close()


# ----------------------------------------------------------------
# APP
# ----------------------------------------------------------------

app = FastAPI(
    title="GeoEngine API",
    description="3D Geospatial Engine — Terrain, OSM, Analysis",
    version="0.1.0",
    debug=config.debug,
    lifespan=lifespan,
    docs_url="/docs"  if config.debug else None,
    redoc_url="/redoc" if config.debug else None,
)

# ---- Middleware ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

# ---- Routers ----
app.include_router(terrain_router)
app.include_router(osm_router)


# ----------------------------------------------------------------
# ENDPOINTS
# ----------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health() -> dict:
    """Health check endpoint."""
    return {
        "status":      "ok",
        "version":     "0.1.0",
        "connections": manager.count,
        "debug":       config.debug,
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint для стримінгу тайлів."""
    await websocket_endpoint(
        websocket=websocket,
        terrain_service=websocket.app.state.terrain,
        analysis_service=websocket.app.state.analysis,
    )
