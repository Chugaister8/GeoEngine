// ============================================================
// GeoEngine — Shared Geo Types
// Базові геопросторові типи, спільні для Python↔JS протоколу
// ============================================================

// ------------------------------------------------------------
// КООРДИНАТИ
// ------------------------------------------------------------

/** Географічна точка у системі WGS84 (EPSG:4326) */
export interface LatLon {
  readonly lat: number  // -90..90
  readonly lon: number  // -180..180
}

/** LatLon + висота над рівнем моря (метри) */
export interface LatLonAlt extends LatLon {
  readonly alt: number
}

/** 2D точка у локальній/екранній системі координат */
export interface Vec2 {
  readonly x: number
  readonly y: number
}

/** 3D точка у світовій системі координат рушія */
export interface Vec3 {
  readonly x: number
  readonly y: number
  readonly z: number
}

/** Quaternion для обертань */
export interface Quat {
  readonly x: number
  readonly y: number
  readonly z: number
  readonly w: number
}

/** 4×4 матриця (column-major, як у WebGPU/GLSL) */
export type Mat4 = readonly [
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
]

// ------------------------------------------------------------
// BOUNDING BOX
// ------------------------------------------------------------

/**
 * Географічний обмежуючий прямокутник (WGS84)
 * Конвенція: west < east, south < north
 * Антимеридіан: west може бути > east (наприклад, 170..-170)
 */
export interface BBox {
  readonly west:  number   // мін. longitude
  readonly south: number   // мін. latitude
  readonly east:  number   // макс. longitude
  readonly north: number   // макс. latitude
}

/** BBox з висотами */
export interface BBox3D extends BBox {
  readonly minAlt: number  // метри
  readonly maxAlt: number  // метри
}

// ------------------------------------------------------------
// TILE ADDRESSING
// ------------------------------------------------------------

/**
 * Адреса тайлу у схемі XYZ (Slippy Map / WebMercator)
 * z=0: весь світ в 1 тайлі
 * z=1: 4 тайли
 * z=N: 4^N тайлів
 */
export interface TileXYZ {
  readonly x: number
  readonly y: number
  readonly z: number  // zoom рівень (0-22)
}

/** Тип тайлової схеми */
export type TileScheme = 'xyz' | 'tms' | 'quadkey'

// ------------------------------------------------------------
// ПРОЕКЦІЇ / CRS
// ------------------------------------------------------------

/**
 * Ідентифікатор системи координат
 * Підтримуємо: EPSG коди та іменовані псевдоніми
 */
export type CRSCode =
  | 'WGS84'          // EPSG:4326 — GPS стандарт
  | 'WebMercator'    // EPSG:3857 — веб-карти
  | 'ECEF'           // Earth-Centered Earth-Fixed (3D глобус)
  | `EPSG:${number}` // Будь-який EPSG (UTM, Lambert, тощо)

/** Точка у довільній системі координат */
export interface ProjectedPoint {
  readonly x: number
  readonly y: number
  readonly crs: CRSCode
}

// ------------------------------------------------------------
// RESOLUTION / LOD
// ------------------------------------------------------------

/** Рівень деталізації (Level of Detail) */
export type LODLevel = 0 | 1 | 2 | 3 | 4 | 5

/**
 * Роздільна здатність у метрах на піксель
 * LOD0 ≈ 0.5–1m, LOD1 ≈ 5–10m, ..., LOD5 ≈ 5–10km
 */
export interface Resolution {
  readonly metersPerPixel: number
  readonly lod: LODLevel
}

// ------------------------------------------------------------
// ELEVATION DATA
// ------------------------------------------------------------

/** Метадані одного тайлу висот */
export interface ElevationTileMeta {
  readonly tile:        TileXYZ
  readonly bbox:        BBox
  readonly resolution:  number        // м/піксель
  readonly minElevation: number       // метри
  readonly maxElevation: number       // метри
  readonly noDataValue:  number       // зазвичай -9999
  readonly source:       DEMSource
}

/** Джерело даних висот */
export type DEMSource =
  | 'srtm30'        // NASA SRTM 30m
  | 'copernicus25'  // ESA Copernicus 25m
  | 'usgs1m'        // USGS 3DEP 1m
  | 'custom'        // Користувацький GeoTIFF

// ------------------------------------------------------------
// CAMERA
// ------------------------------------------------------------

/** Стан камери */
export interface CameraState {
  readonly position:   LatLonAlt
  readonly target:     LatLon
  readonly heading:    number    // градуси від Півночі (0-360)
  readonly pitch:      number    // нахил (-90 вниз, +90 вгору)
  readonly fov:        number    // field of view (градуси)
  readonly near:       number    // near clip plane (метри)
  readonly far:        number    // far clip plane  (метри)
}

// ------------------------------------------------------------
// WEBSOCKET PROTOCOL (Python ↔ JS)
// ------------------------------------------------------------

/** Базовий тип повідомлення по WebSocket */
export interface WSMessage<T extends string, P = unknown> {
  readonly type:      T
  readonly id:        string     // UUID запиту
  readonly timestamp: number     // Unix ms
  readonly payload:   P
}

/** Запит тайлу висот */
export type WSRequestTile = WSMessage<'request_tile', {
  tile:   TileXYZ
  source: DEMSource
}>

/** Відповідь з тайлом (base64 float32 array) */
export type WSResponseTile = WSMessage<'response_tile', {
  tile:    TileXYZ
  width:   number
  height:  number
  data:    string    // base64(Float32Array) висоти в метрах
  meta:    ElevationTileMeta
}>

/** Помилка */
export type WSError = WSMessage<'error', {
  code:    number
  message: string
  detail?: unknown
}>

/** Union всіх серверних повідомлень */
export type WSServerMessage = WSResponseTile | WSError

/** Union всіх клієнтських повідомлень */
export type WSClientMessage = WSRequestTile

// ------------------------------------------------------------
// UTILITY TYPES
// ------------------------------------------------------------

/** Результат операції — або значення або помилка */
export type Result<T, E = Error> =
  | { readonly ok: true;  readonly value: T }
  | { readonly ok: false; readonly error: E }

/** Mutable версія readonly типу */
export type Mutable<T> = { -readonly [P in keyof T]: T[P] }

// ------------------------------------------------------------
// CONSTANTS (як const об'єкт, не enum)
// ------------------------------------------------------------

export const GEO = {
  /** Радіус Землі (метри, середній) */
  EARTH_RADIUS_M: 6_371_000,

  /** WGS84 велика піввісь */
  WGS84_A: 6_378_137.0,

  /** WGS84 мала піввісь */
  WGS84_B: 6_356_752.314245,

  /** WGS84 стиснення */
  WGS84_F: 1 / 298.257223563,

  /** Градусів у радіані */
  DEG_TO_RAD: Math.PI / 180,
  RAD_TO_DEG: 180 / Math.PI,

  /** Розмір тайлу за замовчуванням (пікселів) */
  DEFAULT_TILE_SIZE: 256,

  /** Максимальний WebMercator zoom */
  MAX_ZOOM: 22,
} as const
