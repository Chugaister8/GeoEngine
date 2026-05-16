"""
GeoEngine — WebSocket Handler
Обробка WebSocket з'єднань та повідомлень.

Архітектура:
  ConnectionManager — реєструє/видаляє з'єднання, broadcast
  WSHandler         — обробляє повідомлення одного з'єднання
  TaskQueue         — черга важких задач (DEM fetch, mesh build)

Кожне з'єднання отримує:
  - Унікальний connection_id
  - Власний стан (підписки, кеш, rate limiter)
  - Чергу відповідей (щоб не блокувати event loop)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from .protocol import (
    ClientMessage,
    ErrorCode,
    MessageType,
    PingMessage,
    PongMessage,
    RequestAnalysisMessage,
    RequestTileMessage,
    ResponseTileMessage,
    ResponseTilePayload,
    TileBuffers,
    TileOrigin,
    ServerMessage,
    parse_client_message,
    make_error,
)
from ..config import settings

log: structlog.BoundLogger = structlog.get_logger(__name__)


# ----------------------------------------------------------------
# RATE LIMITER
# ----------------------------------------------------------------

@dataclass
class RateLimiter:
    """
    Простий token bucket rate limiter для одного з'єднання.

    Запобігає спаму запитами тайлів (DEM fetch дорогий).
    """
    max_tokens:    float = 20.0    # максимальний burst
    refill_rate:   float = 5.0     # токенів/секунду
    _tokens:       float = field(init=False)
    _last_refill:  float = field(init=False)

    def __post_init__(self) -> None:
        self._tokens      = self.max_tokens
        self._last_refill = time.monotonic()

    def consume(self, count: float = 1.0) -> bool:
        """
        Спробувати витратити count токенів.

        Returns:
            True якщо дозволено, False якщо rate limit перевищено.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill

        # Поповнити токени
        self._tokens = min(
            self.max_tokens,
            self._tokens + elapsed * self.refill_rate,
        )
        self._last_refill = now

        if self._tokens >= count:
            self._tokens -= count
            return True
        return False


# ----------------------------------------------------------------
# СТАН З'ЄДНАННЯ
# ----------------------------------------------------------------

@dataclass
class ConnectionState:
    """Стан одного WebSocket з'єднання."""
    connection_id: str
    connected_at:  float = field(default_factory=time.monotonic)
    rate_limiter:  RateLimiter = field(default_factory=RateLimiter)
    subscriptions: set[str] = field(default_factory=set)
    request_count: int = 0
    error_count:   int = 0

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.connected_at


# ----------------------------------------------------------------
# CONNECTION MANAGER
# ----------------------------------------------------------------

class ConnectionManager:
    """
    Менеджер всіх активних WebSocket з'єднань.

    Thread-safe через asyncio.Lock.
    Singleton — один на весь FastAPI застосунок.
    """

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._states:      dict[str, ConnectionState] = {}
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> str:
        """
        Прийняти нове з'єднання.

        Returns:
            connection_id — унікальний ідентифікатор з'єднання
        """
        await ws.accept()
        conn_id = str(uuid.uuid4())

        async with self._lock:
            self._connections[conn_id] = ws
            self._states[conn_id] = ConnectionState(connection_id=conn_id)

        log.info(
            "ws.connect",
            conn_id=conn_id[:8],
            total=len(self._connections),
        )
        return conn_id

    async def disconnect(self, conn_id: str) -> None:
        """Закрити та видалити з'єднання."""
        async with self._lock:
            self._connections.pop(conn_id, None)
            state = self._states.pop(conn_id, None)

        if state:
            log.info(
                "ws.disconnect",
                conn_id=conn_id[:8],
                age_s=f"{state.age_s:.1f}",
                requests=state.request_count,
            )

    async def send(self, conn_id: str, message: ServerMessage) -> bool:
        """
        Відправити повідомлення конкретному з'єднанню.

        Returns:
            True якщо успішно, False якщо з'єднання вже закрите.
        """
        ws = self._connections.get(conn_id)
        if ws is None:
            return False

        try:
            await ws.send_text(message.model_dump_json())
            return True
        except Exception as exc:
            log.warning(
                "ws.send.error",
                conn_id=conn_id[:8],
                error=str(exc),
            )
            await self.disconnect(conn_id)
            return False

    async def broadcast(self, message: ServerMessage) -> int:
        """
        Відправити повідомлення всім з'єднанням.

        Returns:
            Кількість успішно отриманих з'єднань.
        """
        if not self._connections:
            return 0

        conn_ids = list(self._connections.keys())
        results  = await asyncio.gather(
            *[self.send(cid, message) for cid in conn_ids],
            return_exceptions=True,
        )
        return sum(1 for r in results if r is True)

    def get_state(self, conn_id: str) -> ConnectionState | None:
        return self._states.get(conn_id)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# Глобальний singleton
manager = ConnectionManager()


# ----------------------------------------------------------------
# WS HANDLER
# ----------------------------------------------------------------

class WSHandler:
    """
    Обробник повідомлень одного WebSocket з'єднання.

    Патерн: Command — кожен тип повідомлення → окремий метод.

    Важкі операції (DEM fetch, mesh build) виконуються
    в ThreadPoolExecutor щоб не блокувати event loop.
    """

    def __init__(self, conn_id: str) -> None:
        self._conn_id  = conn_id
        self._log      = log.bind(conn_id=conn_id[:8])

        # Імпортуємо тут щоб уникнути циклічних залежностей
        from ..services.terrain import TerrainService
        from ..services.analysis import AnalysisService
        self._terrain  = TerrainService()
        self._analysis = AnalysisService()

    async def handle(self, raw: str | bytes) -> ServerMessage | None:
        """
        Обробити одне повідомлення від клієнта.

        Args:
            raw: JSON рядок або bytes

        Returns:
            Відповідь або None (якщо відповідь вже відправлена асинхронно)
        """
        state = manager.get_state(self._conn_id)
        if state is None:
            return None

        state.request_count += 1

        # Парсинг
        try:
            message = parse_client_message(raw)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            state.error_count += 1
            self._log.warning(
                "ws.parse_error",
                error=str(exc)[:200],
            )
            return make_error(
                code=ErrorCode.INVALID_PAYLOAD,
                message=f"Невалідний формат повідомлення: {exc}",
            )

        self._log.debug("ws.message", type=message.type)

        # Rate limiting
        if not state.rate_limiter.consume():
            return make_error(
                code=ErrorCode.RATE_LIMITED,
                message="Занадто багато запитів. Зачекайте секунду.",
                request_id=message.id,
            )

        # Диспетчеризація
        match message.type:
            case MessageType.REQUEST_TILE:
                return await self._handle_tile(message)   # type: ignore[arg-type]
            case MessageType.REQUEST_ANALYSIS:
                return await self._handle_analysis(message)  # type: ignore[arg-type]
            case MessageType.PING:
                return self._handle_ping(message)         # type: ignore[arg-type]
            case MessageType.UNSUBSCRIBE:
                return await self._handle_unsubscribe(message)
            case _:
                return make_error(
                    code=ErrorCode.UNKNOWN_MESSAGE_TYPE,
                    message=f"Невідомий тип повідомлення: {message.type}",
                    request_id=message.id,
                )

    # ---- Обробники ----

    async def _handle_tile(
        self,
        message: RequestTileMessage,
    ) -> ResponseTileMessage | None:
        """
        Обробити запит тайлу.

        Пайплайн:
        1. Завантажити DEM (з кешу або мережі)
        2. Побудувати TerrainMesh
        3. Серіалізувати у base64 буфери
        4. Повернути ResponseTileMessage
        """
        p    = message.payload
        tile = p.tile

        self._log.info(
            "ws.tile.request",
            tile=f"{tile.z}/{tile.x}/{tile.y}",
            source=p.source,
        )

        try:
            # Делегуємо TerrainService
            mesh_data = await self._terrain.get_tile_mesh(
                tile_x=tile.x,
                tile_y=tile.y,
                tile_z=tile.z,
                source=str(p.source),
                max_vertices=p.max_vertices,
                skirt_height_m=p.skirt_height_m,
            )

        except Exception as exc:
            self._log.error(
                "ws.tile.error",
                tile=f"{tile.z}/{tile.x}/{tile.y}",
                error=str(exc),
            )
            return make_error(   # type: ignore[return-value]
                code=ErrorCode.DEM_FETCH_FAILED,
                message=f"Не вдалося отримати DEM: {exc}",
                request_id=message.id,
            )

        response = ResponseTileMessage(
            request_id=message.id,
            payload=ResponseTilePayload(
                lod_level=mesh_data["lod_level"],
                vertex_count=mesh_data["vertex_count"],
                triangle_count=mesh_data["triangle_count"],
                bbox=mesh_data["bbox"],
                origin=TileOrigin(**mesh_data["origin"]),
                buffers=TileBuffers(**mesh_data["buffers"]),
                min_elevation=mesh_data["min_elevation"],
                max_elevation=mesh_data["max_elevation"],
                source=mesh_data["source"],
                resolution_m=mesh_data["resolution_m"],
                memory_bytes=mesh_data["memory_bytes"],
            ),
        )

        self._log.info(
            "ws.tile.done",
            tile=f"{tile.z}/{tile.x}/{tile.y}",
            verts=mesh_data["vertex_count"],
            tris=mesh_data["triangle_count"],
        )

        return response

    async def _handle_analysis(
        self,
        message: RequestAnalysisMessage,
    ) -> ServerMessage:
        """Обробити запит аналізу."""
        from ..services.analysis import AnalysisService
        from .protocol import AnalysisResultMessage, AnalysisResultPayload

        p = message.payload
        self._log.info(
            "ws.analysis.request",
            analyses=p.analyses,
            bbox=f"{p.bbox.west:.2f},{p.bbox.south:.2f},{p.bbox.east:.2f},{p.bbox.north:.2f}",
        )

        try:
            result = await self._analysis.run(
                bbox_west=p.bbox.west,
                bbox_south=p.bbox.south,
                bbox_east=p.bbox.east,
                bbox_north=p.bbox.north,
                analyses=list(p.analyses),
                params={
                    "contour_interval_m": p.contour_interval_m,
                    "viewshed_observer":  p.viewshed_observer,
                    "viewshed_radius_m":  p.viewshed_radius_m,
                    "flood_level_m":      p.flood_level_m,
                    "hillshade_azimuth":  p.hillshade_azimuth,
                    "hillshade_altitude": p.hillshade_altitude,
                },
            )
        except Exception as exc:
            self._log.error("ws.analysis.error", error=str(exc))
            return make_error(
                code=ErrorCode.ANALYSIS_FAILED,
                message=f"Аналіз не вдався: {exc}",
                request_id=message.id,
            )

        return AnalysisResultMessage(
            request_id=message.id,
            payload=AnalysisResultPayload(**result),
        )

    def _handle_ping(self, message: PingMessage) -> PongMessage:
        """Відповісти на ping."""
        return PongMessage(payload={"latency_hint": int(time.time() * 1000)})

    async def _handle_unsubscribe(self, message: Any) -> PongMessage:
        """Відписатися від стріму."""
        state = manager.get_state(self._conn_id)
        if state:
            stream_id = message.payload.get("stream_id", "")
            state.subscriptions.discard(stream_id)
        return PongMessage()


# ----------------------------------------------------------------
# FASTAPI ENDPOINT
# ----------------------------------------------------------------

async def websocket_endpoint(ws: WebSocket) -> None:
    """
    FastAPI WebSocket endpoint.

    Підключити до роутера:
        router.add_api_websocket_route("/ws", websocket_endpoint)

    Lifecycle:
        connect → receive loop → disconnect (clean або exception)
    """
    conn_id = await manager.connect(ws)
    handler = WSHandler(conn_id)
    bound_log = log.bind(conn_id=conn_id[:8])

    try:
        while True:
            # Перевірка стану
            if ws.client_state != WebSocketState.CONNECTED:
                break

            # Отримати повідомлення з таймаутом (keepalive)
            try:
                raw = await asyncio.wait_for(
                    ws.receive_text(),
                    timeout=settings.ws_timeout_s,
                )
            except asyncio.TimeoutError:
                # Клієнт не надсилав нічого — відправляємо ping
                try:
                    await ws.send_text(
                        PongMessage(payload={"type": "keepalive"}).model_dump_json()
                    )
                except Exception:
                    break
                continue

            # Обробка повідомлення
            response = await handler.handle(raw)

            # Відправка відповіді
            if response is not None:
                sent = await manager.send(conn_id, response)
                if not sent:
                    break

    except WebSocketDisconnect:
        bound_log.info("ws.client_disconnect")
    except Exception as exc:
        bound_log.error("ws.unexpected_error", error=str(exc))
        try:
            await manager.send(
                conn_id,
                make_error(
                    code=ErrorCode.INTERNAL_ERROR,
                    message="Внутрішня помилка сервера",
                ),
            )
        except Exception:
            pass
    finally:
        await manager.disconnect(conn_id)
