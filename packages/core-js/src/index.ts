/**
 * GeoEngine Core JS — повний публічний API
 */

// Renderer
export { GeoRenderer }         from "./renderer/GeoRenderer"
export type {
  GeoRendererOptions,
  RenderStats,
  RendererState,
}                              from "./renderer/GeoRenderer"

// Terrain
export { TerrainTile, buildHeightmapGeometry } from "./terrain/TerrainTile"
export type {
  TerrainMeshServerData,
  TileState,
}                              from "./terrain/TerrainTile"

// Quadtree LOD
export { QuadtreeLOD, QuadtreeNode } from "./terrain/Quadtree"
export type {
  LODConfig,
  QuadtreeStats,
  QuadtreeNodeState,
}                              from "./terrain/Quadtree"
export { DEFAULT_LOD_CONFIGS } from "./terrain/Quadtree"

// Geo
export {
  WGS84, DEG2RAD, RAD2DEG,
  llhToECEF, ecefToLLH,
  llhToENU, enuToThreeJS, llhToWorld,
  llhToWebMercator, webMercatorToLLH,
  latLonToTile, tileToLatLonBBox, bboxToTiles,
  tileResolutionM, haversineDistance, bearing,
}                              from "./geo/Coords"
export type {
  ECEF, ENU, WebMercatorPoint,
}                              from "./geo/Coords"

// Utils
export {
  lerp, clamp, clamp01,
  smoothstep, smootherstep,
  easeInOutCubic, easeOutQuart,
  degToRad, radToDeg,
  angleWrap360, angleWrap180,
  vec2, vec2Zero, vec2One,
  vec2Add, vec2Sub, vec2Scale,
  vec2Dot, vec2Len, vec2Norm, vec2Lerp,
  vec3, vec3Zero, vec3One, vec3Up,
  vec3Forward, vec3Right,
  vec3Add, vec3Sub, vec3Scale, vec3Neg,
  vec3Dot, vec3Cross, vec3Len, vec3LenSq,
  vec3Norm, vec3Lerp, vec3Dist, vec3DistSq,
  vec3Reflect,
  mat4Identity, mat4Multiply,
  mat4Translation, mat4Scale, mat4RotationY,
  mat4Perspective, mat4LookAt,
  mat4TransformVec3,
  LRUCache,
}                              from "./utils/index"
export type { Vec2, Vec3, Mat4 } from "./utils/math"
export { ObjectPool, Float32ArrayPool } from "./utils/pool"
