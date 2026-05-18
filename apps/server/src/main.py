"""
GeoEngine — FastAPI Application
Головний entry point сервера.

Запуск:
    uvicorn apps.server.src.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health                     — health check
    WS   /ws                         — WebSocket стримінг тайлів

    GET  /api/terrain/sources        — список DEM джерел
    GET  /api/terrain/tile/{z}/{x}/{y}/meta
    GET  /api/terrain/tile/{z}/{x}/{y}.png
    POST /api/terrain/elevation      — висоти для точок
    POST /api/terrain/mesh           — 3D mesh для bbox
    GET  /api/terrain/cache/stats
    DELETE /api/terrain/cache

    POST /api/osm/buildings          — будівлі OSM
    POST /api/osm/roads              — дороги OSM
    POST /api/osm/full               — всі OSM дані
    GET  /api/osm/cache/clear

    POST /api/sim/ballistics         — траєкторія снаряду
    POST /api/sim/ballistics/table   — таблиця стрільби
    POST /api/sim/fire               — симуляція вогню
    GET  /api/sim/ballistics/presets — пресети снарядів
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .config  import config
from .api.terrain    import router as terrain_router, set_terrain_service
from .api.osm        import router as osm_router
from .api.simulation import router as sim_router
from .services.terrain  import TerrainService
from .services.analysis import AnalysisService
from .ws.handler    import websocket_endpoint, manager

from geoengine.utils.logging import configure_logging

# ── Ініціалізація логування ────────────────────────────────────
configure_logging(
    level=config.log_level,
    json_logs=config.json_logs,
    service="geoengine-server",
)
log: structlog.BoundLogger = structlog.get_logger(__name__)


# ── LIFESPAN ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup → yield → Shutdown.

    Startup:
      - Ініціалізуємо TerrainService (ThreadPool + кеш)
      - Ініціалізуємо AnalysisService
      - Зберігаємо у app.state для доступу через DI

    Shutdown:
      - Graceful shutdown ExecutorPool
      - Закриваємо всі WS з'єднання
    """
    t_start = time.perf_counter()
    log.info(
        "server.startup",
        version="0.1.0",
        debug=config.debug,
        host=config.host,
        port=config.port,
        workers=config.terrain_workers,
    )

    # ── Terrain Service ─────────────────────────────────────
    terrain_svc = await TerrainService.create(
        dem_cache_dir=config.dem_cache_dir,
        max_workers=config.terrain_workers,
        api_keys=config.dem_api_keys,
    )

    # ── Analysis Service ────────────────────────────────────
    analysis_svc = AnalysisService(
        terrain_service=terrain_svc,
        max_workers=config.analysis_workers,
    )

    # ── DI реєстрація ───────────────────────────────────────
    app.state.terrain  = terrain_svc
    app.state.analysis = analysis_svc
    set_terrain_service(terrain_svc)

    elapsed = (time.perf_counter() - t_start) * 1000
    log.info(
        "server.ready",
        startup_ms=round(elapsed, 1),
        cache_dir=config.dem_cache_dir,
    )

    yield  # ← сервер працює

    # ── Shutdown ────────────────────────────────────────────
    log.info(
        "server.shutdown",
        active_connections=manager.count,
    )
    await terrain_svc.close()
    await analysis_svc.close()
    log.info("server.stopped")


# ── APP ────────────────────────────────────────────────────────

app = FastAPI(
    title       = "GeoEngine API",
    description = (
        "3D Geospatial Engine — Terrain, OSM, Analysis, Simulation.\n\n"
        "WebSocket: `ws://localhost:8000/ws`\n"
        "Source: https://github.com/your-org/geoengine"
    ),
    version     = "0.1.0",
    debug       = config.debug,
    lifespan    = lifespan,
    # Swagger UI тільки в debug режимі
    docs_url    = "/docs"  if config.debug else None,
    redoc_url   = "/redoc" if config.debug else None,
    openapi_url = "/openapi.json" if config.debug else None,
)


# ── MIDDLEWARE ─────────────────────────────────────────────────

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)

# GZip для великих відповідей (mesh data)
app.add_middleware(
    GZipMiddleware,
    minimum_size=1024,   # стискаємо відповіді > 1KB
)


# ── REQUEST TIMING MIDDLEWARE ──────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Логування часу виконання кожного запиту."""
    t0       = time.perf_counter()
    response = await call_next(request)
    elapsed  = (time.perf_counter() - t0) * 1000

    log.debug(
        "http.request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        ms=round(elapsed, 1),
    )
    response.headers["X-Process-Time"] = f"{elapsed:.1f}ms"
    return response


# ── EXCEPTION HANDLERS ─────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "type": "ValueError"},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    log.error(
        "http.unhandled_error",
        path=request.url.path,
        error=str(exc)[:200],
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type":   type(exc).__name__,
        },
    )


# ── ROUTERS ────────────────────────────────────────────────────

app.include_router(terrain_router)   # /api/terrain/*
app.include_router(osm_router)       # /api/osm/*
app.include_router(sim_router)       # /api/sim/*


# ── CORE ENDPOINTS ─────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health() -> dict:
    """
    Health check endpoint.

    Відповідає:
      status:      "ok"
      version:     "0.1.0"
      connections: кількість активних WS з'єднань
      debug:       чи увімкнений debug режим
      uptime_s:    час роботи сервера (секунди)
    """
    return {
        "status":      "ok",
        "version":     "0.1.0",
        "connections": manager.count,
        "debug":       config.debug,
        "uptime_s":    round(time.perf_counter(), 1),
    }


@app.get("/api/info", tags=["system"])
async def api_info() -> dict:
    """
    Повна інформація про сервер та доступні endpoints.
    """
    terrain_stats = None
    try:
        terrain_stats = await app.state.terrain.cache_stats()
    except Exception:
        pass

    return {
        "version":   "0.1.0",
        "debug":     config.debug,
        "endpoints": {
            "terrain":    "/api/terrain",
            "osm":        "/api/osm",
            "simulation": "/api/sim",
            "websocket":  "/ws",
            "health":     "/health",
        },
        "capabilities": [
            "terrain_mesh",
            "terrain_png",
            "elevation_query",
            "osm_buildings",
            "osm_roads",
            "slope_analysis",
            "hillshade",
            "contours",
            "ballistics",
            "fire_simulation",
        ],
        "cache":     terrain_stats,
        "ws_connections": manager.count,
    }


# ── WEBSOCKET ──────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint для real-time стримінгу тайлів.

    Протокол:
      Client → Server:
        {"type":"ping",           "id":"...", "timestamp":0, "payload":{}}
        {"type":"request_tile",   "id":"...", "payload":{"tile":{x,y,z},...}}
        {"type":"request_analysis","id":"...", "payload":{"bbox":{},...}}
        {"type":"camera_update",  "id":"...", "payload":{"lat":0,"lon":0,...}}

      Server → Client:
        {"type":"connected",    "payload":{"session_id":"..."}}
        {"type":"pong",         "request_id":"..."}
        {"type":"response_tile","request_id":"...", "payload":{...}}
        {"type":"error",        "payload":{"code":1001,"message":"..."}}

    Rate limit: 10 запитів/секунду (burst: 20)
    """
    await websocket_endpoint(
        websocket=websocket,
        terrain_service=websocket.app.state.terrain,
        analysis_service=websocket.app.state.analysis,
    )
