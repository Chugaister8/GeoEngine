"""
GeoEngine — WebSocket Handler
Обробка WS з'єднань та диспетчеризація повідомлень.

Архітектура:
  ConnectionManager  — реєстр активних з'єднань
  RateLimiter        — token bucket per connection
  WSHandler          — диспетчер повідомлень (match/case)
  websocket_endpoint — FastAPI WebSocket endpoint
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict
from typing import Any

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from .protocol import (
    WSClientMessage, WSRequestTile, WSRequestAnalysis,
    WSCameraUpdate, WSPing,
    parse_client_message,
    make_pong, make_error, make_connected,
    WSError,
)

log: structlog.BoundLogger = structlog.get_logger(__name__)

# ----------------------------------------------------------------
# ERROR CODES
# ----------------------------------------------------------------

WS_ERRORS = {
    "INVALID_MESSAGE":   1001,
    "INVALID_TILE":      1002,
    "SOURCE_NOT_FOUND":  1003,
    "PROCESSING_FAILED": 1004,
    "RATE_LIMITED":      4001,
    "SERVER_ERROR":      5000,
}

# ----------------------------------------------------------------
# RATE LIMITER (Token Bucket)
# ----------------------------------------------------------------

class RateLimiter:
    """
    Token bucket rate limiter per connection.
    Дозволяє burst запитів але обмежує середній rate.
    """

    def __init__(
        self,
        rate:     float = 10.0,   # запитів/секунду
        capacity: float = 20.0,   # максимальний burst
    ) -> None:
        self._rate     = rate
        self._capacity = capacity
        self._tokens:  dict[str, float] = defaultdict(lambda: capacity)
        self._last:    dict[str, float] = defaultdict(time.monotonic)

    def allow(self, conn_id: str) -> bool:
        """Чи дозволений запит для conn_id?"""
        now     = time.monotonic()
        elapsed = now - self._last[conn_id]
        self._last[conn_id] = now

        # Поповнити токени за час що минув
        self._tokens[conn_id] = min(
            self._capacity,
            self._tokens[conn_id] + elapsed * self._rate,
        )

        if self._tokens[conn_id] >= 1.0:
            self._tokens[conn_id] -= 1.0
            return True
        return False

    def remove(self, conn_id: str) -> None:
        self._tokens.pop(conn_id, None)
        self._last.pop(conn_id, None)


# ----------------------------------------------------------------
# CONNECTION MANAGER
# ----------------------------------------------------------------

class ConnectionManager:
    """Реєстр активних WebSocket з'єднань."""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> str:
        """Прийняти нове з'єднання. Повертає session_id."""
        await websocket.accept()
        session_id = str(uuid.uuid4())
        self._connections[session_id] = websocket
        log.info("ws.connect", session_id=session_id[:8],
                 total=len(self._connections))
        return session_id

    def disconnect(self, session_id: str) -> None:
        self._connections.pop(session_id, None)
        log.info("ws.disconnect", session_id=session_id[:8],
                 total=len(self._connections))

    async def send(self, session_id: str, data: dict) -> bool:
        """Відправити повідомлення конкретному клієнту."""
        ws = self._connections.get(session_id)
        if ws is None:
            return False
        try:
            await ws.send_json(data)
            return True
        except Exception as exc:
            log.warning("ws.send_error", session_id=session_id[:8],
                        error=str(exc)[:80])
            return False

    async def broadcast(self, data: dict) -> int:
        """Broadcast всім клієнтам. Повертає кількість успішних."""
        sent = 0
        for sid in list(self._connections.keys()):
            if await self.send(sid, data):
                sent += 1
        return sent

    @property
    def count(self) -> int:
        return len(self._connections)


# ---- Глобальні інстанси ----
manager     = ConnectionManager()
rate_limiter = RateLimiter(rate=10.0, capacity=20.0)


# ----------------------------------------------------------------
# WS HANDLER
# ----------------------------------------------------------------

class WSHandler:
    """
    Диспетчер WebSocket повідомлень.
    Використовує match/case для маршрутизації типів.
    """

    def __init__(
        self,
        session_id:      str,
        terrain_service: Any,
        analysis_service: Any,
    ) -> None:
        self._sid             = session_id
        self._terrain_svc     = terrain_service
        self._analysis_svc    = analysis_service

    async def handle(self, raw: str) -> dict | None:
        """
        Обробити одне WS повідомлення.
        Повертає відповідь або None (якщо немає відповіді).
        """
        # 1. Парсинг JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.debug("ws.parse_error", error=str(exc)[:60])
            return make_error(
                WS_ERRORS["INVALID_MESSAGE"],
                f"Invalid JSON: {exc}",
            )

        # 2. Rate limiting
        if not rate_limiter.allow(self._sid):
            log.warning("ws.rate_limited", session_id=self._sid[:8])
            return make_error(
                WS_ERRORS["RATE_LIMITED"],
                "Rate limit exceeded. Max 10 req/sec.",
                request_id=data.get("id"),
            )

        # 3. Парсинг типу повідомлення
        msg = parse_client_message(data)
        if msg is None:
            return make_error(
                WS_ERRORS["INVALID_MESSAGE"],
                f"Unknown message type: {data.get('type', 'none')}",
                request_id=data.get("id"),
            )

        # 4. Диспетчеризація
        try:
            return await self._dispatch(msg)
        except Exception as exc:
            log.error("ws.handler_error", error=str(exc), exc_info=True)
            return make_error(
                WS_ERRORS["SERVER_ERROR"],
                "Internal server error",
                request_id=msg.id,
                details=str(exc)[:200],
            )

    async def _dispatch(self, msg: WSClientMessage) -> dict | None:
        """Маршрутизація за типом повідомлення."""
        match msg:

            case WSPing():
                recv_ts = time.time() * 1000
                return make_pong(
                    request_id=msg.id,
                    latency_ms=recv_ts - msg.timestamp,
                )

            case WSRequestTile():
                return await self._handle_tile(msg)

            case WSRequestAnalysis():
                return await self._handle_analysis(msg)

            case WSCameraUpdate():
                # Camera update — просто логуємо, не відповідаємо
                log.debug(
                    "ws.camera_update",
                    lat=round(msg.payload.lat, 4),
                    lon=round(msg.payload.lon, 4),
                    alt=round(msg.payload.alt, 0),
                )
                return None

            case _:
                return make_error(
                    WS_ERRORS["INVALID_MESSAGE"],
                    f"Unhandled type: {msg.type}",
                    request_id=msg.id,
                )

    async def _handle_tile(self, msg: WSRequestTile) -> dict:
        """Обробити запит тайлу."""
        p = msg.payload

        log.debug(
            "ws.tile_request",
            tile=f"{p.tile.z}/{p.tile.x}/{p.tile.y}",
            source=p.source,
            max_verts=p.max_vertices,
        )

        t0 = time.perf_counter()

        try:
            mesh_data = await self._terrain_svc.get_tile_mesh(
                x=p.tile.x,
                y=p.tile.y,
                z=p.tile.z,
                source=p.source,
                max_vertices=p.max_vertices,
                skirt_height_m=p.skirt_height_m,
            )
        except ValueError as exc:
            return make_error(
                WS_ERRORS["INVALID_TILE"],
                str(exc),
                request_id=msg.id,
            )
        except Exception as exc:
            log.error("ws.tile_error",
                      tile=f"{p.tile.z}/{p.tile.x}/{p.tile.y}",
                      error=str(exc))
            return make_error(
                WS_ERRORS["PROCESSING_FAILED"],
                "Failed to generate terrain mesh",
                request_id=msg.id,
                details=str(exc)[:200],
            )

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(
            "ws.tile_response",
            tile=f"{p.tile.z}/{p.tile.x}/{p.tile.y}",
            verts=mesh_data.get("vertex_count", 0),
            ms=round(elapsed, 1),
        )

        return {
            "type":       "response_tile",
            "id":         str(uuid.uuid4()),
            "timestamp":  time.time() * 1000,
            "request_id": msg.id,
            "payload":    mesh_data,
        }

    async def _handle_analysis(self, msg: WSRequestAnalysis) -> dict:
        """Обробити запит аналізу."""
        p = msg.payload

        log.debug(
            "ws.analysis_request",
            analyses=p.analyses,
            bbox=str(p.bbox),
        )

        results = []
        for analysis_type in p.analyses:
            try:
                result = await self._analysis_svc.compute(
                    bbox=p.bbox,
                    analysis_type=analysis_type,
                    source=p.source,
                    options=p.options,
                )
                results.append(result)
            except Exception as exc:
                log.warning(
                    "ws.analysis_error",
                    type=analysis_type,
                    error=str(exc)[:80],
                )

        if not results:
            return make_error(
                WS_ERRORS["PROCESSING_FAILED"],
                "All analyses failed",
                request_id=msg.id,
            )

        return {
            "type":       "analysis_result",
            "id":         str(uuid.uuid4()),
            "timestamp":  time.time() * 1000,
            "request_id": msg.id,
            "payload": {
                "results": results,
                "bbox":    p.bbox.model_dump(),
            },
        }


# ----------------------------------------------------------------
# FASTAPI ENDPOINT
# ----------------------------------------------------------------

async def websocket_endpoint(
    websocket:        WebSocket,
    terrain_service:  Any,
    analysis_service: Any,
) -> None:
    """
    FastAPI WebSocket endpoint.

    Lifecycle:
      connect → send connected → recv loop → disconnect
    """
    session_id = await manager.connect(websocket)

    # Відправити connected повідомлення
    await websocket.send_json(make_connected(session_id))

    handler = WSHandler(
        session_id=session_id,
        terrain_service=terrain_service,
        analysis_service=analysis_service,
    )

    try:
        while True:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=120.0,   # 2 хвилини timeout
                )
            except asyncio.TimeoutError:
                # Ping для перевірки чи жива сесія
                await websocket.send_json({"type": "ping", "id": "server-keepalive", "timestamp": time.time()*1000, "payload": {}})
                continue

            response = await handler.handle(raw)
            if response is not None:
                await websocket.send_json(response)

    except WebSocketDisconnect:
        log.info("ws.client_disconnected", session_id=session_id[:8])
    except Exception as exc:
        log.error("ws.error", session_id=session_id[:8], error=str(exc))
        try:
            await websocket.send_json(
                make_error(WS_ERRORS["SERVER_ERROR"], str(exc)[:200])
            )
        except Exception:
            pass
    finally:
        manager.disconnect(session_id)
        rate_limiter.remove(session_id)
