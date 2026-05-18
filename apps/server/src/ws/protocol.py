"""
GeoEngine — WebSocket Protocol
Pydantic моделі для всіх WS повідомлень.

Протокол (Client → Server):
  ping              { type, id, timestamp, payload:{} }
  request_tile      { ..., payload:{tile, source, max_vertices, skirt_height_m} }
  request_analysis  { ..., payload:{bbox, analyses, source, options} }
  camera_update     { ..., payload:{lat,lon,alt,heading,pitch} }

Протокол (Server → Client):
  pong              { type, id, request_id, timestamp, payload:{latency_ms} }
  response_tile     { ..., payload:{tile, lod_level, vertex_count, buffers,...} }
  analysis_result   { ..., payload:{analysis_type, result_type, data,...} }
  error             { ..., payload:{code, message} }
  connected         { ..., payload:{session_id, server_version} }
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator


# ----------------------------------------------------------------
# BASE
# ----------------------------------------------------------------

class WSBase(BaseModel):
    """Базова модель WS повідомлення."""
    type:      str
    id:        str   = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=lambda: time.time() * 1000)


# ----------------------------------------------------------------
# CLIENT → SERVER
# ----------------------------------------------------------------

class TileXYZ(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    z: int = Field(ge=0, le=22)


class BBoxModel(BaseModel):
    west:  float = Field(ge=-180, le=180)
    south: float = Field(ge=-90,  le=90)
    east:  float = Field(ge=-180, le=180)
    north: float = Field(ge=-90,  le=90)

    @field_validator("north")
    @classmethod
    def north_gt_south(cls, v: float, info: Any) -> float:
        if "south" in (info.data or {}) and v <= info.data["south"]:
            raise ValueError("north must be > south")
        return v


class PingPayload(BaseModel):
    pass


class RequestTilePayload(BaseModel):
    tile:           TileXYZ
    source:         str     = "terrarium"
    max_vertices:   int     = Field(default=65_536, ge=64, le=262_144)
    skirt_height_m: float   = Field(default=200.0, ge=0.0)
    lod_level:      int     = Field(default=0, ge=0, le=5)


class RequestAnalysisPayload(BaseModel):
    bbox:      BBoxModel
    analyses:  list[str]  = Field(default=["slope"])
    source:    str        = "terrarium"
    options:   dict[str, Any] = Field(default_factory=dict)


class CameraUpdatePayload(BaseModel):
    lat:     float
    lon:     float
    alt:     float
    heading: float = 0.0
    pitch:   float = -30.0
    fov:     float = 60.0


class WSPing(WSBase):
    type:    Literal["ping"]
    payload: PingPayload = Field(default_factory=PingPayload)


class WSRequestTile(WSBase):
    type:    Literal["request_tile"]
    payload: RequestTilePayload


class WSRequestAnalysis(WSBase):
    type:    Literal["request_analysis"]
    payload: RequestAnalysisPayload


class WSCameraUpdate(WSBase):
    type:    Literal["camera_update"]
    payload: CameraUpdatePayload


# Union для парсингу будь-якого клієнтського повідомлення
WSClientMessage = Union[
    WSPing,
    WSRequestTile,
    WSRequestAnalysis,
    WSCameraUpdate,
]


# ----------------------------------------------------------------
# SERVER → CLIENT
# ----------------------------------------------------------------

class WSPong(WSBase):
    type:       Literal["pong"] = "pong"
    request_id: str

    class Config:
        extra = "allow"


class TerrainMeshBuffers(BaseModel):
    vertices: str    # base64 Float32Array
    indices:  str    # base64 Uint32Array
    uvs:      str    # base64 Float32Array
    normals:  str    # base64 Float32Array


class ResponseTilePayload(BaseModel):
    tile:           TileXYZ
    lod_level:      int
    vertex_count:   int
    triangle_count: int
    memory_bytes:   int
    bbox:           BBoxModel
    origin:         dict[str, float]   # {lat, lon, alt}
    min_elevation:  float
    max_elevation:  float
    source:         str
    buffers:        TerrainMeshBuffers
    normal_map:     dict[str, Any] | None = None


class WSResponseTile(WSBase):
    type:       Literal["response_tile"] = "response_tile"
    request_id: str
    payload:    ResponseTilePayload


class AnalysisResultPayload(BaseModel):
    analysis_type: str
    result_type:   str
    bbox:          BBoxModel
    data:          str | None = None     # base64
    width:         int | None = None
    height:        int | None = None
    min_val:       float | None = None
    max_val:       float | None = None
    geojson:       dict | None = None
    profile:       dict | None = None


class WSAnalysisResult(WSBase):
    type:       Literal["analysis_result"] = "analysis_result"
    request_id: str
    payload:    AnalysisResultPayload


class ErrorPayload(BaseModel):
    code:    int
    message: str
    details: str | None = None


class WSError(WSBase):
    type:       Literal["error"] = "error"
    request_id: str | None = None
    payload:    ErrorPayload


class ConnectedPayload(BaseModel):
    session_id:     str
    server_version: str = "0.1.0"
    capabilities:   list[str] = Field(default_factory=list)


class WSConnected(WSBase):
    type:    Literal["connected"] = "connected"
    payload: ConnectedPayload


# ----------------------------------------------------------------
# ПАРСИНГ
# ----------------------------------------------------------------

_CLIENT_TYPES: dict[str, type] = {
    "ping":             WSPing,
    "request_tile":     WSRequestTile,
    "request_analysis": WSRequestAnalysis,
    "camera_update":    WSCameraUpdate,
}


def parse_client_message(
    data: dict[str, Any],
) -> WSClientMessage | None:
    """
    Розпарсити JSON dict у відповідну Pydantic модель.
    Повертає None якщо type невідомий або дані невалідні.
    """
    msg_type = data.get("type", "")
    model_cls = _CLIENT_TYPES.get(msg_type)
    if model_cls is None:
        return None
    try:
        return model_cls.model_validate(data)
    except Exception:
        return None


def make_pong(request_id: str, latency_ms: float = 0.0) -> dict:
    return WSPong(
        request_id=request_id,
        payload={"latency_ms": latency_ms},
    ).model_dump()


def make_error(
    code:       int,
    message:    str,
    request_id: str | None = None,
    details:    str | None = None,
) -> dict:
    return WSError(
        request_id=request_id,
        payload=ErrorPayload(code=code, message=message, details=details),
    ).model_dump()


def make_connected(session_id: str) -> dict:
    return WSConnected(
        payload=ConnectedPayload(
            session_id=session_id,
            capabilities=["terrain", "analysis", "osm"],
        )
    ).model_dump()
