"""
GeoEngine — WebSocket Protocol
Типи повідомлень між Python сервером та JS клієнтом.

Всі повідомлення — JSON об'єкти з полями:
  type:      рядковий ідентифікатор типу
  id:        UUID запиту (для match request→response)
  timestamp: Unix мілісекунди
  payload:   тіло повідомлення (залежить від type)

Клієнт → Сервер:
  request_tile     — запит DEM тайлу
  request_analysis — запит аналітики (slope, viewshed, тощо)
  subscribe_stream — підписка на live дані
  ping             — keepalive

Сервер → Клієнт:
  response_tile    — DEM меш у відповідь на request_tile
  analysis_result  — результат аналізу
  stream_update    — live оновлення (IoT, погода)
  error            — помилка
  pong             — відповідь на ping
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator


# ----------------------------------------------------------------
# БАЗОВІ ТИПИ
# ----------------------------------------------------------------

class MessageType(StrEnum):
    # Клієнт → Сервер
    REQUEST_TILE     = "request_tile"
    REQUEST_ANALYSIS = "request_analysis"
    SUBSCRIBE_STREAM = "subscribe_stream"
    UNSUBSCRIBE      = "unsubscribe"
    PING             = "ping"

    # Сервер → Клієнт
    RESPONSE_TILE    = "response_tile"
    ANALYSIS_RESULT  = "analysis_result"
    STREAM_UPDATE    = "stream_update"
    ERROR            = "error"
    PONG             = "pong"


class TileXYZ(BaseModel):
    """XYZ тайл адреса."""
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    z: int = Field(ge=0, le=22)

    model_config = {"frozen": True}


class BBoxModel(BaseModel):
    """Географічний BBox."""
    west:  float = Field(ge=-180, le=180)
    south: float = Field(ge=-90,  le=90)
    east:  float = Field(ge=-180, le=180)
    north: float = Field(ge=-90,  le=90)

    model_config = {"frozen": True}

    @field_validator("north")
    @classmethod
    def north_gt_south(cls, north: float, info: Any) -> float:
        south = info.data.get("south", -90)
        if north <= south:
            raise ValueError(f"north={north} має бути > south={south}")
        return north


class DEMSourceEnum(StrEnum):
    SRTM30       = "srtm30"
    SRTM90       = "srtm90"
    COPERNICUS25 = "copernicus25"
    TERRARIUM    = "terrarium"
    CUSTOM       = "custom"


# ----------------------------------------------------------------
# БАЗОВИЙ КЛАС ПОВІДОМЛЕННЯ
# ----------------------------------------------------------------

class BaseMessage(BaseModel):
    """Базове повідомлення протоколу."""
    type:      str
    id:        str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))

    model_config = {"frozen": True}


# ----------------------------------------------------------------
# КЛІЄНТ → СЕРВЕР
# ----------------------------------------------------------------

class RequestTilePayload(BaseModel):
    """Payload запиту тайлу."""
    tile:   TileXYZ
    source: DEMSourceEnum = DEMSourceEnum.COPERNICUS25
    # Опційні параметри
    include_normals:  bool  = True
    include_analysis: bool  = False   # додати slope/hillshade у відповідь
    skirt_height_m:   float = Field(default=200.0, ge=0, le=10_000)
    max_vertices:     int   = Field(default=65_536, ge=64, le=262_144)

    model_config = {"frozen": True}


class RequestAnalysisPayload(BaseModel):
    """Payload запиту аналізу."""
    bbox:     BBoxModel
    analyses: list[Literal[
        "slope", "aspect", "hillshade",
        "contours", "viewshed", "flood",
    ]]
    # Параметри специфічних аналізів
    contour_interval_m: float = 100.0
    viewshed_observer:  tuple[float, float, float] | None = None  # lat, lon, height_m
    viewshed_radius_m:  float = 5000.0
    flood_level_m:      float = 0.0
    hillshade_azimuth:  float = 315.0
    hillshade_altitude: float = 45.0

    model_config = {"frozen": True}


class SubscribeStreamPayload(BaseModel):
    """Підписка на потік даних."""
    stream_type: Literal["weather", "iot", "gps_track", "satellite"]
    bbox:        BBoxModel | None = None
    interval_s:  float = Field(default=10.0, ge=1, le=3600)

    model_config = {"frozen": True}


# ---- Повідомлення клієнта ----

class RequestTileMessage(BaseMessage):
    type:    Literal[MessageType.REQUEST_TILE] = MessageType.REQUEST_TILE
    payload: RequestTilePayload


class RequestAnalysisMessage(BaseMessage):
    type:    Literal[MessageType.REQUEST_ANALYSIS] = MessageType.REQUEST_ANALYSIS
    payload: RequestAnalysisPayload


class SubscribeStreamMessage(BaseMessage):
    type:    Literal[MessageType.SUBSCRIBE_STREAM] = MessageType.SUBSCRIBE_STREAM
    payload: SubscribeStreamPayload


class UnsubscribeMessage(BaseMessage):
    type: Literal[MessageType.UNSUBSCRIBE] = MessageType.UNSUBSCRIBE
    payload: dict[str, str] = Field(default_factory=dict)  # {"stream_id": "..."}


class PingMessage(BaseMessage):
    type:    Literal[MessageType.PING] = MessageType.PING
    payload: dict = Field(default_factory=dict)


# Union всіх клієнтських повідомлень
ClientMessage = Annotated[
    Union[
        RequestTileMessage,
        RequestAnalysisMessage,
        SubscribeStreamMessage,
        UnsubscribeMessage,
        PingMessage,
    ],
    Field(discriminator="type"),
]


# ----------------------------------------------------------------
# СЕРВЕР → КЛІЄНТ
# ----------------------------------------------------------------

class TileOrigin(BaseModel):
    """Origin координати для ENU системи."""
    lat: float
    lon: float
    alt: float = 0.0

    model_config = {"frozen": True}


class TileBuffers(BaseModel):
    """Base64-кодовані GPU буфери."""
    vertices: str   # base64(Float32Array) (N,3) XYZ метри ENU
    indices:  str   # base64(Uint32Array)  (M,3) трикутники
    uvs:      str   # base64(Float32Array) (N,2) текстурні координати
    normals:  str   # base64(Float32Array) (N,3) нормалі

    model_config = {"frozen": True}


class ResponseTilePayload(BaseModel):
    """Payload відповіді з тайлом."""
    type:           Literal["terrain_mesh"] = "terrain_mesh"
    lod_level:      int
    vertex_count:   int
    triangle_count: int
    bbox:           list[float]   # [west, south, east, north]
    origin:         TileOrigin
    buffers:        TileBuffers
    # Метадані
    min_elevation:  float
    max_elevation:  float
    source:         str
    resolution_m:   float
    memory_bytes:   int

    model_config = {"frozen": True}


class AnalysisResultPayload(BaseModel):
    """Payload результату аналізу."""
    analysis_type: str
    bbox:          list[float]
    # Результат залежно від типу
    # slope/aspect/hillshade: base64 float32 raster
    # contours: GeoJSON FeatureCollection
    # viewshed: base64 bool raster
    result_type:   Literal["raster", "geojson", "value"]
    data:          str | dict   # base64 або GeoJSON
    width:         int | None = None
    height:        int | None = None
    metadata:      dict = Field(default_factory=dict)

    model_config = {"frozen": True}


class ErrorPayload(BaseModel):
    """Payload помилки."""
    code:    int
    message: str
    detail:  Any = None

    model_config = {"frozen": True}


class ErrorCode:
    """Коди помилок."""
    UNKNOWN_MESSAGE_TYPE = 1001
    INVALID_PAYLOAD      = 1002
    TILE_NOT_FOUND       = 2001
    DEM_FETCH_FAILED     = 2002
    ANALYSIS_FAILED      = 3001
    RATE_LIMITED         = 4001
    INTERNAL_ERROR       = 5000


# ---- Повідомлення сервера ----

class ResponseTileMessage(BaseMessage):
    type:        Literal[MessageType.RESPONSE_TILE] = MessageType.RESPONSE_TILE
    request_id:  str    # id оригінального RequestTileMessage
    payload:     ResponseTilePayload


class AnalysisResultMessage(BaseMessage):
    type:       Literal[MessageType.ANALYSIS_RESULT] = MessageType.ANALYSIS_RESULT
    request_id: str
    payload:    AnalysisResultPayload


class ErrorMessage(BaseMessage):
    type:       Literal[MessageType.ERROR] = MessageType.ERROR
    request_id: str | None = None
    payload:    ErrorPayload


class PongMessage(BaseMessage):
    type:    Literal[MessageType.PONG] = MessageType.PONG
    payload: dict = Field(default_factory=dict)


# Union всіх серверних повідомлень
ServerMessage = Union[
    ResponseTileMessage,
    AnalysisResultMessage,
    ErrorMessage,
    PongMessage,
]


# ----------------------------------------------------------------
# ПАРСЕР ВХІДНИХ ПОВІДОМЛЕНЬ
# ----------------------------------------------------------------

from pydantic import TypeAdapter

_client_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


def parse_client_message(raw: str | bytes | dict) -> ClientMessage:
    """
    Розпарсити повідомлення від клієнта.

    Args:
        raw: JSON рядок, bytes або вже розпарсений dict

    Returns:
        Типізоване повідомлення клієнта

    Raises:
        pydantic.ValidationError: невалідний формат
        ValueError:               невідомий type
    """
    import json
    if isinstance(raw, (str, bytes)):
        data = json.loads(raw)
    else:
        data = raw

    return _client_adapter.validate_python(data)


def make_error(
    code:       int,
    message:    str,
    request_id: str | None = None,
    detail:     Any = None,
) -> ErrorMessage:
    """Зручний конструктор повідомлення про помилку."""
    return ErrorMessage(
        request_id=request_id,
        payload=ErrorPayload(code=code, message=message, detail=detail),
    )
