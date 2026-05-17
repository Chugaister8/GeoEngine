"""
GeoEngine — OSM Fetcher
Завантаження даних OpenStreetMap через Overpass API.

Overpass API — read-only API для OSM що дозволяє
робити складні просторові запити за bbox, тегами, типами.

Підтримує:
  - Будівлі (building=*)
  - Дороги (highway=*)
  - Природні об'єкти (natural=*, landuse=*)
  - Водойми (water=*, waterway=*)
  - Адміністративні межі (boundary=administrative)
  - Довільні Overpass QL запити
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal

import httpx
import structlog

from ..geo.bbox import BBox

log: structlog.BoundLogger = structlog.get_logger(__name__)

# ----------------------------------------------------------------
# КОНСТАНТИ
# ----------------------------------------------------------------

OVERPASS_ENDPOINTS: Final[list[str]] = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

REQUEST_TIMEOUT:  Final[float] = 120.0   # Overpass може бути повільним
MAX_RETRIES:      Final[int]   = 3
RETRY_DELAY:      Final[float] = 5.0
CACHE_TTL_S:      Final[int]   = 86_400  # 24 години


# ----------------------------------------------------------------
# ТИПИ ДАНИХ
# ----------------------------------------------------------------

class OSMElementType(StrEnum):
    NODE     = "node"
    WAY      = "way"
    RELATION = "relation"


@dataclass(frozen=True, slots=True)
class OSMNode:
    """OSM вузол (точка)."""
    id:   int
    lat:  float
    lon:  float
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OSMWay:
    """OSM шлях (лінія або полігон)."""
    id:       int
    node_ids: tuple[int, ...]
    tags:     dict[str, str] = field(default_factory=dict)
    # Координати заповнюються після resolve_nodes
    coords:   tuple[tuple[float, float], ...] = field(default_factory=tuple)

    @property
    def is_closed(self) -> bool:
        """Чи є шлях замкненим (полігон)."""
        return (
            len(self.node_ids) >= 4
            and self.node_ids[0] == self.node_ids[-1]
        )

    @property
    def is_building(self) -> bool:
        return "building" in self.tags

    @property
    def is_highway(self) -> bool:
        return "highway" in self.tags

    @property
    def is_water(self) -> bool:
        return (
            "water" in self.tags
            or self.tags.get("natural") in ("water", "wetland")
            or "waterway" in self.tags
        )


@dataclass(frozen=True, slots=True)
class OSMRelation:
    """OSM відношення (складні полігони, маршрути)."""
    id:      int
    members: tuple[dict[str, Any], ...]
    tags:    dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class OSMData:
    """
    Результат Overpass запиту.
    Містить nodes, ways, relations + індекс nodes за id.
    """
    bbox:      BBox
    nodes:     list[OSMNode]     = field(default_factory=list)
    ways:      list[OSMWay]      = field(default_factory=list)
    relations: list[OSMRelation] = field(default_factory=list)

    # Індекс nodes за id (для resolve_nodes)
    _node_index: dict[int, OSMNode] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        self._node_index = {n.id: n for n in self.nodes}

    def get_node(self, node_id: int) -> OSMNode | None:
        return self._node_index.get(node_id)

    def buildings(self) -> list[OSMWay]:
        return [w for w in self.ways if w.is_building]

    def highways(self) -> list[OSMWay]:
        return [w for w in self.ways if w.is_highway]

    def water_bodies(self) -> list[OSMWay]:
        return [w for w in self.ways if w.is_water]

    def ways_with_tag(self, key: str, value: str | None = None) -> list[OSMWay]:
        if value is None:
            return [w for w in self.ways if key in w.tags]
        return [w for w in self.ways if w.tags.get(key) == value]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def way_count(self) -> int:
        return len(self.ways)

    @property
    def building_count(self) -> int:
        return len(self.buildings())

    def __repr__(self) -> str:
        return (
            f"OSMData("
            f"nodes={self.node_count}, "
            f"ways={self.way_count}, "
            f"buildings={self.building_count}, "
            f"bbox={self.bbox})"
        )


# ----------------------------------------------------------------
# OVERPASS QUERY BUILDER
# ----------------------------------------------------------------

class OverpassQuery:
    """
    Будівник Overpass QL запитів.

    Overpass QL — мова запитів для OSM даних.
    Документація: https://wiki.openstreetmap.org/wiki/Overpass_API/Language_Guide

    Usage:
        query = (OverpassQuery()
            .bbox(bbox)
            .buildings()
            .highways(["primary", "secondary"])
            .timeout(60)
            .build())
    """

    def __init__(self) -> None:
        self._bbox:     BBox | None = None
        self._timeout:  int  = 60
        self._parts:    list[str] = []

    def bbox(self, bbox: BBox) -> "OverpassQuery":
        """Встановити географічний bbox для запиту."""
        self._bbox = bbox
        return self

    def timeout(self, seconds: int) -> "OverpassQuery":
        """Таймаут Overpass запиту (секунди)."""
        self._timeout = seconds
        return self

    def buildings(self) -> "OverpassQuery":
        """Додати запит будівель."""
        self._parts.append('way["building"]')
        return self

    def highways(
        self,
        types: list[str] | None = None,
    ) -> "OverpassQuery":
        """
        Додати запит доріг.

        Args:
            types: список типів (motorway, primary, secondary, ...)
                   None = всі типи
        """
        if types:
            values = "|".join(types)
            self._parts.append(f'way["highway"~"^({values})$"]')
        else:
            self._parts.append('way["highway"]')
        return self

    def natural(
        self,
        values: list[str] | None = None,
    ) -> "OverpassQuery":
        """Природні об'єкти (wood, water, peak, ...)."""
        if values:
            v = "|".join(values)
            self._parts.append(f'way["natural"~"^({v})$"]')
            self._parts.append(f'node["natural"~"^({v})$"]')
        else:
            self._parts.append('way["natural"]')
            self._parts.append('node["natural"]')
        return self

    def water(self) -> "OverpassQuery":
        """Водойми та річки."""
        self._parts.extend([
            'way["water"]',
            'way["waterway"~"^(river|stream|canal|drain)$"]',
            'way["natural"="water"]',
            'relation["water"]',
        ])
        return self

    def landuse(
        self,
        values: list[str] | None = None,
    ) -> "OverpassQuery":
        """Землекористування (forest, residential, industrial, ...)."""
        if values:
            v = "|".join(values)
            self._parts.append(f'way["landuse"~"^({v})$"]')
        else:
            self._parts.append('way["landuse"]')
        return self

    def amenities(
        self,
        values: list[str] | None = None,
    ) -> "OverpassQuery":
        """Зручності (school, hospital, restaurant, ...)."""
        if values:
            v = "|".join(values)
            self._parts.append(f'node["amenity"~"^({v})$"]')
            self._parts.append(f'way["amenity"~"^({v})$"]')
        else:
            self._parts.append('node["amenity"]')
        return self

    def custom(self, overpass_ql: str) -> "OverpassQuery":
        """Довільний Overpass QL фрагмент."""
        self._parts.append(overpass_ql.strip())
        return self

    def build(self) -> str:
        """
        Побудувати фінальний Overpass QL рядок.

        Returns:
            Повний Overpass QL запит готовий для API

        Raises:
            ValueError: якщо bbox не встановлений
        """
        if self._bbox is None:
            raise ValueError("BBox не встановлений. Викличте .bbox(bbox) спочатку.")

        if not self._parts:
            raise ValueError("Немає частин запиту. Додайте .buildings(), .highways() тощо.")

        # Overpass bbox формат: south,west,north,east
        b = self._bbox
        bbox_str = f"{b.south},{b.west},{b.north},{b.east}"

        # Формуємо union всіх частин
        union_parts = "\n  ".join(
            f"{part}({bbox_str});"
            for part in self._parts
        )

        return (
            f"[out:json][timeout:{self._timeout}];\n"
            f"(\n"
            f"  {union_parts}\n"
            f");\n"
            f"out body;\n"
            f">;\n"
            f"out skel qt;"
        )


# ----------------------------------------------------------------
# OVERPASS FETCHER
# ----------------------------------------------------------------

class OverpassFetcher:
    """
    Async клієнт для Overpass API.

    Особливості:
    - Автоматичне переключення між endpoints (failover)
    - Disk cache з TTL (щоб не дублювати запити)
    - Retry з exponential backoff
    - Rate limiting (Overpass має ліміти)

    Usage:
        fetcher = OverpassFetcher(cache_dir="~/.geoengine/osm_cache")
        data = await fetcher.fetch(
            bbox=BBox(22, 47, 24, 49),
            query=OverpassQuery().bbox(bbox).buildings().highways(),
        )
    """

    def __init__(
        self,
        cache_dir:    str | Path = Path.home() / ".geoengine" / "osm_cache",
        cache_ttl_s:  int = CACHE_TTL_S,
        endpoints:    list[str] | None = None,
    ) -> None:
        self._cache_dir  = Path(cache_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl  = cache_ttl_s
        self._endpoints  = endpoints or OVERPASS_ENDPOINTS
        self._current_ep = 0   # поточний endpoint індекс

    # ---- Публічний API ----

    async def fetch(
        self,
        bbox:  BBox,
        query: "OverpassQuery | str",
    ) -> OSMData:
        """
        Завантажити OSM дані для bbox.

        Args:
            bbox:  географічний BBox
            query: OverpassQuery об'єкт або готовий QL рядок

        Returns:
            OSMData з nodes, ways, relations

        Raises:
            OverpassError: помилка API
            OverpassTimeoutError: таймаут запиту
        """
        if isinstance(query, OverpassQuery):
            ql = query.build()
        else:
            ql = query

        cache_key  = self._cache_key(ql)
        cache_path = self._cache_dir / f"{cache_key}.json"

        # Перевірити кеш
        if self._is_cache_valid(cache_path):
            log.debug("osm.fetch.cache_hit", key=cache_key[:12])
            return self._load_from_cache(cache_path, bbox)

        log.info(
            "osm.fetch.start",
            bbox=str(bbox),
            buildings="building" in ql,
            highways="highway" in ql,
        )

        # Завантажити з API
        raw_json = await self._fetch_with_retry(ql)

        # Зберегти в кеш
        cache_path.write_text(
            json.dumps(raw_json, ensure_ascii=False),
            encoding="utf-8",
        )

        osm_data = self._parse_response(raw_json, bbox)
        log.info(
            "osm.fetch.done",
            nodes=osm_data.node_count,
            ways=osm_data.way_count,
            buildings=osm_data.building_count,
        )
        return osm_data

    async def fetch_buildings(self, bbox: BBox) -> OSMData:
        """Зручний метод: тільки будівлі."""
        query = OverpassQuery().bbox(bbox).buildings().timeout(60)
        return await self.fetch(bbox=bbox, query=query)

    async def fetch_roads(
        self,
        bbox:        BBox,
        road_types:  list[str] | None = None,
    ) -> OSMData:
        """Зручний метод: дороги."""
        types = road_types or [
            "motorway", "trunk", "primary", "secondary",
            "tertiary", "residential", "service",
        ]
        query = OverpassQuery().bbox(bbox).highways(types).timeout(60)
        return await self.fetch(bbox=bbox, query=query)

    async def fetch_full(self, bbox: BBox) -> OSMData:
        """
        Повне завантаження: будівлі + дороги + вода + природа.
        Для великих bbox може бути повільним (30-120 сек).
        """
        query = (
            OverpassQuery()
            .bbox(bbox)
            .buildings()
            .highways([
                "motorway", "trunk", "primary", "secondary",
                "tertiary", "residential",
            ])
            .water()
            .natural(["wood", "forest", "water", "peak"])
            .landuse(["forest", "residential", "industrial", "farmland"])
            .timeout(120)
        )
        return await self.fetch(bbox=bbox, query=query)

    def clear_cache(self) -> int:
        """Очистити OSM кеш. Повертає кількість видалених файлів."""
        count = 0
        for f in self._cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        log.info("osm.cache.cleared", files=count)
        return count

    # ---- Приватні методи ----

    async def _fetch_with_retry(self, ql: str) -> dict:
        """Fetch з retry та endpoint failover."""
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES * len(self._endpoints)):
            endpoint = self._endpoints[self._current_ep % len(self._endpoints)]

            try:
                log.debug(
                    "osm.http.request",
                    endpoint=endpoint,
                    attempt=attempt + 1,
                )

                async with httpx.AsyncClient(
                    timeout=REQUEST_TIMEOUT,
                ) as client:
                    response = await client.post(
                        endpoint,
                        data={"data": ql},
                        headers={"Accept": "application/json"},
                    )

                if response.status_code == 200:
                    data = response.json()
                    if "elements" not in data:
                        raise OverpassError(
                            f"Невалідна відповідь Overpass: {str(data)[:200]}"
                        )
                    return data

                if response.status_code == 429:
                    wait = (2 ** (attempt % MAX_RETRIES)) * RETRY_DELAY
                    log.warning("osm.http.rate_limit", wait=wait)
                    await asyncio.sleep(wait)
                    continue

                if response.status_code == 504:
                    raise OverpassTimeoutError(
                        f"Overpass timeout (504): спробуй зменшити bbox або запит"
                    )

                raise OverpassError(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = exc
                # Перейти на наступний endpoint
                self._current_ep += 1
                wait = RETRY_DELAY * (attempt + 1)
                log.warning(
                    "osm.http.error",
                    endpoint=endpoint,
                    error=str(exc)[:100],
                    next_wait=wait,
                )
                await asyncio.sleep(wait)

        raise OverpassError(
            f"Всі спроби вичерпані після {MAX_RETRIES} ретраїв: {last_error}"
        )

    @staticmethod
    def _parse_response(raw: dict, bbox: BBox) -> OSMData:
        """
        Розпарсити JSON відповідь Overpass API.

        Формат відповіді:
        {
          "elements": [
            {"type": "node", "id": 123, "lat": 48.0, "lon": 23.0, "tags": {}},
            {"type": "way",  "id": 456, "nodes": [123, ...], "tags": {}},
            ...
          ]
        }
        """
        nodes:     list[OSMNode]     = []
        ways:      list[OSMWay]      = []
        relations: list[OSMRelation] = []

        for elem in raw.get("elements", []):
            elem_type = elem.get("type")
            elem_id   = int(elem.get("id", 0))
            tags      = elem.get("tags", {})

            if elem_type == "node":
                lat = elem.get("lat")
                lon = elem.get("lon")
                if lat is not None and lon is not None:
                    nodes.append(OSMNode(
                        id=elem_id,
                        lat=float(lat),
                        lon=float(lon),
                        tags=tags,
                    ))

            elif elem_type == "way":
                node_ids = tuple(int(n) for n in elem.get("nodes", []))
                ways.append(OSMWay(
                    id=elem_id,
                    node_ids=node_ids,
                    tags=tags,
                ))

            elif elem_type == "relation":
                members = tuple(elem.get("members", []))
                relations.append(OSMRelation(
                    id=elem_id,
                    members=members,
                    tags=tags,
                ))

        osm_data = OSMData(
            bbox=bbox,
            nodes=nodes,
            ways=ways,
            relations=relations,
        )

        # Resolve координати ways
        _resolve_way_coords(osm_data)

        return osm_data

    def _is_cache_valid(self, path: Path) -> bool:
        """Чи валідний кеш файл (існує і не протермінований)."""
        if not path.exists():
            return False
        age_s = time.time() - path.stat().st_mtime
        return age_s < self._cache_ttl

    def _load_from_cache(self, path: Path, bbox: BBox) -> OSMData:
        """Завантажити OSM дані з кешу."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        return self._parse_response(raw, bbox)

    @staticmethod
    def _cache_key(ql: str) -> str:
        """SHA256 хеш запиту як ключ кешу."""
        return hashlib.sha256(ql.encode()).hexdigest()[:24]


# ----------------------------------------------------------------
# RESOLVE WAY COORDINATES
# ----------------------------------------------------------------

def _resolve_way_coords(osm_data: OSMData) -> None:
    """
    Заповнити coords у всіх ways на основі node_ids.
    Модифікує osm_data.ways in-place (через заміну елементів).
    """
    resolved: list[OSMWay] = []

    for way in osm_data.ways:
        coords: list[tuple[float, float]] = []
        for node_id in way.node_ids:
            node = osm_data.get_node(node_id)
            if node is not None:
                coords.append((node.lat, node.lon))

        resolved.append(OSMWay(
            id=way.id,
            node_ids=way.node_ids,
            tags=way.tags,
            coords=tuple(coords),
        ))

    osm_data.ways = resolved


# ----------------------------------------------------------------
# EXCEPTIONS
# ----------------------------------------------------------------

class OverpassError(Exception):
    """Загальна помилка Overpass API."""
    pass


class OverpassTimeoutError(OverpassError):
    """Timeout Overpass запиту."""
    pass
