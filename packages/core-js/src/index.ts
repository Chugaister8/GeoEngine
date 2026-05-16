/**
 * GeoEngine Core JS — публічний API
 */

// Renderer
export { GeoRenderer }         from './renderer/GeoRenderer'
export type { GeoRendererOptions, RenderStats, RendererState } from './renderer/GeoRenderer'

// Terrain
export { TerrainTile, buildHeightmapGeometry } from './terrain/TerrainTile'
export type { TerrainMeshServerData, TileState } from './terrain/TerrainTile'

// Quadtree LOD
export { QuadtreeLOD, QuadtreeNode }            from './terrain/Quadtree'
export type { LODConfig, QuadtreeStats, QuadtreeNodeState } from './terrain/Quadtree'
export { DEFAULT_LOD_CONFIGS }                  from './terrain/Quadtree'

// Geo utilities
export {
  WGS84, DEG2RAD, RAD2DEG,
  llhToECEF, ecefToLLH,
  llhToENU, enuToThreeJS, llhToWorld,
  llhToWebMercator, webMercatorToLLH,
  latLonToTile, tileToLatLonBBox, bboxToTiles,
  tileResolutionM, haversineDistance, bearing,
} from './geo/Coords'
export type { ECEF, ENU, WebMercatorPoint } from './geo/Coords'
