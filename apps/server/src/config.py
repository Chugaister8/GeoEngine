"""
GeoEngine — Server Configuration
"""

from __future__ import annotations

from geoengine.utils.config import BaseGeoConfig


class ServerConfig(BaseGeoConfig):
    """
    Конфігурація FastAPI сервера.
    Розширює BaseGeoConfig серверними параметрами.
    """

    # WebSocket
    ws_max_connections:  int   = 100
    ws_rate_limit:       float = 10.0    # req/sec per connection
    ws_rate_burst:       float = 20.0    # burst capacity

    # Terrain
    max_mesh_vertices:   int   = 262_144
    max_bbox_area_deg2:  float = 16.0    # 4°×4°

    # OSM
    max_osm_bbox_deg2:   float = 1.0     # 1°×1°

    # Timeouts
    dem_fetch_timeout:   float = 120.0   # секунди
    osm_fetch_timeout:   float = 120.0

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls()


# Глобальний інстанс
config = ServerConfig()
