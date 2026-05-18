/**
 * GeoEngine — GeoRenderer Types
 * Публічні типи для GeoRenderer.
 */

import type { QuadtreeStats } from '../terrain/Quadtree'

export type RendererState =
  | 'idle'
  | 'initializing'
  | 'running'
  | 'paused'
  | 'error'

export interface RenderStats {
  fps:          number
  frameTime:    number
  visibleTiles: number
  totalTiles:   number
  triangles:    number
  drawCalls:    number
  quadtree:     QuadtreeStats
}

export interface GeoRendererOptions {
  canvas:         string | HTMLCanvasElement
  renderer?:      'webgpu' | 'webgl2' | 'auto'
  initialCamera?: { lat: number; lon: number; altitude: number }
  serverUrl?:     string
  originLat?:     number
  originLon?:     number
  maxLODZoom?:    number
  debug?:         boolean
}
