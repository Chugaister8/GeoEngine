/**
 * GeoEngine — Coordinate Utilities (JavaScript/TypeScript)
 * Дзеркало Python coords.py але для браузера/Node.js.
 *
 * Без зовнішніх залежностей — чистий TypeScript.
 * Tree-shakeable: імпортуй тільки те що потрібно.
 */

import type { LatLon, LatLonAlt, Vec3, BBox, TileXYZ, CRSCode } from '../../shared-types/src/geo'

// ----------------------------------------------------------------
// WGS84 константи
// ----------------------------------------------------------------

export const WGS84 = {
  A:      6_378_137.0,                    // велика піввісь (м)
  B:      6_356_752.314245,               // мала піввісь  (м)
  F:      1.0 / 298.257223563,            // стиснення
  E2:     1.0 - (6_356_752.314245 / 6_378_137.0) ** 2, // ексцентриситет²
  RADIUS: 6_371_000.0,                    // середній радіус (м)
} as const

export const DEG2RAD = Math.PI / 180.0
export const RAD2DEG = 180.0 / Math.PI

// ----------------------------------------------------------------
// ТИПИ (локальні розширення)
// ----------------------------------------------------------------

export interface ECEF {
  readonly x: number
  readonly y: number
  readonly z: number
}

export interface ENU {
  readonly east:  number
  readonly north: number
  readonly up:    number
}

export interface WebMercatorPoint {
  readonly x: number  // метри
  readonly y: number  // метри
}

// ----------------------------------------------------------------
// LLH ↔ ECEF
// ----------------------------------------------------------------

/**
 * Geographic WGS84 → Earth-Centered Earth-Fixed (ECEF).
 * Точність: субміліметрова.
 */
export function llhToECEF(lat: number, lon: number, alt = 0): ECEF {
  const latR = lat * DEG2RAD
  const lonR = lon * DEG2RAD

  const sinLat = Math.sin(latR)
  const cosLat = Math.cos(latR)
  const sinLon = Math.sin(lonR)
  const cosLon = Math.cos(lonR)

  // Радіус кривини у першому вертикалі
  const N = WGS84.A / Math.sqrt(1.0 - WGS84.E2 * sinLat * sinLat)

  return {
    x: (N + alt) * cosLat * cosLon,
    y: (N + alt) * cosLat * sinLon,
    z: (N * (1.0 - WGS84.E2) + alt) * sinLat,
  }
}

/**
 * ECEF → Geographic WGS84.
 * Алгоритм Bowring — збіжність < 3 ітерацій.
 */
export function ecefToLLH(x: number, y: number, z: number): LatLonAlt {
  const p = Math.sqrt(x * x + y * y)
  const lon = Math.atan2(y, x)

  let lat = Math.atan2(z, p * (1.0 - WGS84.E2))

  for (let i = 0; i < 10; i++) {
    const sinLat = Math.sin(lat)
    const N = WGS84.A / Math.sqrt(1.0 - WGS84.E2 * sinLat * sinLat)
    const latNew = Math.atan2(z + WGS84.E2 * N * sinLat, p)
    if (Math.abs(latNew - lat) < 1e-12) break
    lat = latNew
  }

  const sinLat = Math.sin(lat)
  const N = WGS84.A / Math.sqrt(1.0 - WGS84.E2 * sinLat * sinLat)
  const cosLat = Math.cos(lat)
  const alt = Math.abs(cosLat) > 1e-10
    ? p / cosLat - N
    : Math.abs(z) / Math.abs(sinLat) - N * (1.0 - WGS84.E2)

  return {
    lat: lat * RAD2DEG,
    lon: lon * RAD2DEG,
    alt,
  }
}

// ----------------------------------------------------------------
// LLH ↔ ENU (локальна система рушія)
// ----------------------------------------------------------------

/**
 * LLH → ENU відносно origin.
 *
 * ENU = East/North/Up у метрах від origin.
 * Це world-space рушія для тайлів до ~500км.
 */
export function llhToENU(
  lat: number, lon: number, alt: number,
  originLat: number, originLon: number, originAlt = 0,
): ENU {
  const p  = llhToECEF(lat, lon, alt)
  const o  = llhToECEF(originLat, originLon, originAlt)

  const dx = p.x - o.x
  const dy = p.y - o.y
  const dz = p.z - o.z

  const latR = originLat * DEG2RAD
  const lonR = originLon * DEG2RAD

  const sinLat = Math.sin(latR)
  const cosLat = Math.cos(latR)
  const sinLon = Math.sin(lonR)
  const cosLon = Math.cos(lonR)

  return {
    east:  -sinLon * dx + cosLon * dy,
    north: -sinLat * cosLon * dx - sinLat * sinLon * dy + cosLat * dz,
    up:     cosLat * cosLon * dx + cosLat * sinLon * dy + sinLat * dz,
  }
}

/**
 * ENU → Three.js Vec3: { x: East, y: Up, z: -North }
 * Three.js конвенція: Y вгору, Z до глядача.
 */
export function enuToThreeJS(enu: ENU): Vec3 {
  return {
    x:  enu.east,
    y:  enu.up,
    z: -enu.north,
  }
}

/**
 * Швидка LLH → Three.js Vec3 (через ENU).
 * Оптимізована версія для масових обчислень в LOD.
 */
export function llhToWorld(
  lat: number, lon: number, alt: number,
  originLat: number, originLon: number, originAlt = 0,
): Vec3 {
  const enu = llhToENU(lat, lon, alt, originLat, originLon, originAlt)
  return enuToThreeJS(enu)
}

// ----------------------------------------------------------------
// WebMercator
// ----------------------------------------------------------------

/**
 * LLH WGS84 → WebMercator (EPSG:3857, метри).
 * Стандарт для веб-тайлів (Mapbox, MapTiler, OSM).
 */
export function llhToWebMercator(lat: number, lon: number): WebMercatorPoint {
  const latClamped = Math.max(-85.051129, Math.min(85.051129, lat))
  return {
    x: WGS84.A * lon * DEG2RAD,
    y: WGS84.A * Math.log(
      Math.tan(Math.PI / 4 + latClamped * DEG2RAD / 2)
    ),
  }
}

/**
 * WebMercator → LLH WGS84.
 */
export function webMercatorToLLH(x: number, y: number): LatLon {
  return {
    lat: (2 * Math.atan(Math.exp(y / WGS84.A)) - Math.PI / 2) * RAD2DEG,
    lon: (x / WGS84.A) * RAD2DEG,
  }
}

// ----------------------------------------------------------------
// TILE ADDRESSING
// ----------------------------------------------------------------

/**
 * LatLon → XYZ тайл адреса на заданому zoom.
 */
export function latLonToTile(lat: number, lon: number, zoom: number): TileXYZ {
  const n = 1 << zoom   // 2^zoom
  const latClamped = Math.max(-85.051129, Math.min(85.051129, lat))
  const latR = latClamped * DEG2RAD

  const x = Math.floor((lon + 180) / 360 * n)
  const y = Math.floor(
    (1 - Math.log(Math.tan(latR) + 1 / Math.cos(latR)) / Math.PI) / 2 * n
  )

  return {
    x: Math.max(0, Math.min(n - 1, x)),
    y: Math.max(0, Math.min(n - 1, y)),
    z: zoom,
  }
}

/**
 * XYZ тайл → BBox (WGS84).
 */
export function tileToLatLonBBox(tile: TileXYZ): BBox {
  const n = 1 << tile.z

  const west  =  tile.x / n * 360 - 180
  const east  = (tile.x + 1) / n * 360 - 180
  const north = _mercYToLat(tile.y,     n)
  const south = _mercYToLat(tile.y + 1, n)

  return { west, south, east, north }
}

/**
 * BBox → список XYZ тайлів що покривають її на zoom.
 */
export function bboxToTiles(bbox: BBox, zoom: number): TileXYZ[] {
  const tl = latLonToTile(bbox.north, bbox.west, zoom)
  const br = latLonToTile(bbox.south, bbox.east, zoom)
  const tiles: TileXYZ[] = []
  for (let y = tl.y; y <= br.y; y++) {
    for (let x = tl.x; x <= br.x; x++) {
      tiles.push({ x, y, z: zoom })
    }
  }
  return tiles
}

/**
 * Роздільна здатність тайлу (метри/піксель).
 * Залежить від широти (WebMercator стискає на полюсах).
 */
export function tileResolutionM(tile: TileXYZ, tileSize = 256): number {
  const bbox = tileToLatLonBBox(tile)
  const centerLat = (bbox.south + bbox.north) / 2
  return (
    2 * Math.PI * WGS84.A
    * Math.cos(centerLat * DEG2RAD)
    / (tileSize * (1 << tile.z))
  )
}

// ----------------------------------------------------------------
// ВІДСТАНІ
// ----------------------------------------------------------------

/**
 * Відстань Гаверсинуса між двома точками (метри).
 * Точність ~0.5% (сфера). Для геодезії — використовуй Vincenty.
 */
export function haversineDistance(
  lat1: number, lon1: number,
  lat2: number, lon2: number,
): number {
  const dLat = (lat2 - lat1) * DEG2RAD
  const dLon = (lon2 - lon1) * DEG2RAD
  const a1   = lat1 * DEG2RAD
  const a2   = lat2 * DEG2RAD

  const sinDLat = Math.sin(dLat / 2)
  const sinDLon = Math.sin(dLon / 2)

  const h = sinDLat * sinDLat + Math.cos(a1) * Math.cos(a2) * sinDLon * sinDLon
  return 2 * WGS84.A * Math.asin(Math.sqrt(h))
}

/**
 * Початковий азимут від (lat1,lon1) до (lat2,lon2).
 * Повертає градуси [0..360) від Північ за годинниковою.
 */
export function bearing(
  lat1: number, lon1: number,
  lat2: number, lon2: number,
): number {
  const lat1R = lat1 * DEG2RAD
  const lat2R = lat2 * DEG2RAD
  const dLon  = (lon2 - lon1) * DEG2RAD

  const x = Math.sin(dLon) * Math.cos(lat2R)
  const y = Math.cos(lat1R) * Math.sin(lat2R)
    - Math.sin(lat1R) * Math.cos(lat2R) * Math.cos(dLon)

  return ((Math.atan2(x, y) * RAD2DEG) + 360) % 360
}

// ----------------------------------------------------------------
// ПРИВАТНІ ХЕЛПЕРИ
// ----------------------------------------------------------------

function _mercYToLat(y: number, n: number): number {
  return Math.atan(Math.sinh(Math.PI * (1 - 2 * y / n))) * RAD2DEG
}
