/**
 * GeoEngine — Shared TypeScript Types
 * Типи що розділяються між core-js, viewer та server протоколом.
 *
 * Цей файл є єдиним джерелом правди для:
 * - Географічних примітивів (BBox, TileXYZ, LatLon)
 * - WebSocket протоколу (запити/відповіді)
 * - LOD конфігурації
 * - CRS кодів та DEM джерел
 */

// ================================================================
// ГЕОГРАФІЧНІ ПРИМІТИВИ
// ================================================================

export interface LatLon {
  lat: number
  lon: number
}

export interface LatLonAlt extends LatLon {
  alt: number
}

export interface BBox {
  west:  number
  south: number
  east:  number
  north: number
}

export interface TileXYZ {
  x: number
  y: number
  z: number
}

export interface Vec3 {
  x: number
  y: number
  z: number
}

export interface Vec2 {
  x: number
  y: number
}

// ================================================================
// LOD
// ================================================================

export type LODLevel = 0 | 1 | 2 | 3 | 4 | 5

export interface LODConfig {
  level:        LODLevel
  minZoom:      number
  maxZoom:      number
  maxVertices:  number
  distanceM:    number   // відстань камери при якій активується
  morphRange:   number   // діапазон morph transition (метри)
}

export const DEFAULT_LOD_LEVELS: LODLevel[] = [0, 1, 2, 3, 4, 5]

// ================================================================
// CRS ТА DEM ДЖЕРЕЛА
// ================================================================

export type CRSCode =
  | "EPSG:4326"    // WGS84 geographic
  | "EPSG:3857"    // Web Mercator
  | "EPSG:32636"   // UTM zone 36N (Ukraine)
  | "EPSG:32637"   // UTM zone 37N
  | string

export type DEMSource =
  | "copernicus25"
  | "srtm30"
  | "terrarium"
  | "usgs1m"
  | string

// ================================================================
// CAMERA STATE
// ================================================================

export interface CameraState {
  lat:     number
  lon:     number
  alt:     number    // метри
  heading: number    // градуси від Півночі [0..360)
  pitch:   number    // градуси [-90..90]
  fov:     number    // degrees
  near:    number    // метри
  far:     number    // метри
}

// ================================================================
// WEBSOCKET ПРОТОКОЛ
// ================================================================

// ---- Базові типи ----

export type WSMessageType =
  | "ping"
  | "pong"
  | "request_tile"
  | "response_tile"
  | "request_analysis"
  | "analysis_result"
  | "camera_update"
  | "scene_update"
  | "error"
  | "connected"
  | "disconnected"

export interface WSBaseMessage {
  type:      WSMessageType
  id:        string       // UUID запиту
  timestamp: number       // Unix мс
}

// ---- Client → Server ----

export interface WSPing extends WSBaseMessage {
  type:    "ping"
  payload: Record<string, never>
}

export interface WSRequestTile extends WSBaseMessage {
  type: "request_tile"
  payload: {
    tile:            TileXYZ
    source:          DEMSource
    max_vertices?:   number      // default 65536
    skirt_height_m?: number      // default 200
    lod_level?:      LODLevel
  }
}

export interface WSRequestAnalysis extends WSBaseMessage {
  type: "request_analysis"
  payload: {
    bbox:      BBox
    analyses:  AnalysisType[]
    source?:   DEMSource
    options?:  AnalysisOptions
  }
}

export interface WSCameraUpdate extends WSBaseMessage {
  type: "camera_update"
  payload: CameraState
}

// ---- Server → Client ----

export interface WSPong extends WSBaseMessage {
  type:       "pong"
  request_id: string
  payload:    { latency_ms?: number }
}

export interface TerrainMeshBuffers {
  vertices: string    // base64 Float32Array (N×3)
  indices:  string    // base64 Uint32Array  (M×3)
  uvs:      string    // base64 Float32Array (N×2)
  normals:  string    // base64 Float32Array (N×3)
}

export interface WSResponseTile extends WSBaseMessage {
  type:       "response_tile"
  request_id: string
  payload: {
    tile:           TileXYZ
    lod_level:      LODLevel
    vertex_count:   number
    triangle_count: number
    memory_bytes:   number
    bbox:           BBox
    origin: {
      lat: number
      lon: number
      alt: number
    }
    min_elevation:  number
    max_elevation:  number
    source:         DEMSource
    buffers:        TerrainMeshBuffers
    // Опційно: normal map
    normal_map?: {
      data:   string   // base64 uint8 (H×W×3)
      width:  number
      height: number
    }
  }
}

export interface WSAnalysisResult extends WSBaseMessage {
  type:       "analysis_result"
  request_id: string
  payload: {
    analysis_type: AnalysisType
    result_type:   "raster" | "vector" | "profile"
    bbox:          BBox
    // Для raster
    data?:    string    // base64 Float32Array
    width?:   number
    height?:  number
    min_val?: number
    max_val?: number
    // Для vector (GeoJSON)
    geojson?: object
    // Для profile
    profile?: ElevationProfile
  }
}

export interface WSError extends WSBaseMessage {
  type:       "error"
  request_id?: string
  payload: {
    code:     number
    message:  string
    details?: string
  }
}

export interface WSConnected extends WSBaseMessage {
  type: "connected"
  payload: {
    session_id:    string
    server_version: string
    capabilities:  string[]
  }
}

// ---- Union types ----

export type WSClientMessage =
  | WSPing
  | WSRequestTile
  | WSRequestAnalysis
  | WSCameraUpdate

export type WSServerMessage =
  | WSPong
  | WSResponseTile
  | WSAnalysisResult
  | WSError
  | WSConnected

// ================================================================
// ANALYSIS TYPES
// ================================================================

export type AnalysisType =
  | "slope"
  | "aspect"
  | "hillshade"
  | "contours"
  | "viewshed"
  | "profile"
  | "flood"

export interface AnalysisOptions {
  // Slope / Aspect
  z_factor?:       number
  // Hillshade
  azimuth?:        number    // degrees [0..360]
  altitude?:       number    // degrees [0..90]
  // Contours
  interval?:       number    // метри
  base?:           number
  // Viewshed
  observer_height?: number   // метри над рельєфом
  max_distance?:   number    // метри
  // Profile
  n_points?:       number
}

export interface ElevationProfile {
  distances:  number[]    // метри від початку
  elevations: number[]    // метри
  lats:       number[]
  lons:       number[]
  total_length_m: number
  min_elevation:  number
  max_elevation:  number
}

// ================================================================
// SCENE TYPES
// ================================================================

export interface SceneLayerInfo {
  id:           string
  name:         string
  visible:      boolean
  opacity:      number
  render_order: number
  node_count:   number
}

export interface SceneInfo {
  name:        string
  origin:      LatLonAlt
  camera:      CameraState
  layers:      SceneLayerInfo[]
  node_count:  number
  bbox?:       BBox
  time_of_day: number
}

// ================================================================
// RENDERER TYPES
// ================================================================

export type RendererMode = "webgpu" | "webgl2"

export interface QuadtreeStats {
  total:    number
  leaves:   number
  culled:   number
  maxDepth: number
}

export interface RenderStats {
  fps:          number
  frameTime:    number
  visibleTiles: number
  totalTiles:   number
  triangles:    number
  drawCalls:    number
  quadtree:     QuadtreeStats
}

export type RendererState =
  | "idle"
  | "initializing"
  | "running"
  | "paused"
  | "error"

// ================================================================
// ERROR CODES
// ================================================================

export const WS_ERROR_CODES = {
  INVALID_MESSAGE:    1001,
  INVALID_TILE:       1002,
  SOURCE_NOT_FOUND:   1003,
  PROCESSING_FAILED:  1004,
  RATE_LIMITED:       4001,
  UNAUTHORIZED:       4003,
  SERVER_ERROR:       5000,
} as const

export type WSErrorCode = typeof WS_ERROR_CODES[keyof typeof WS_ERROR_CODES]
