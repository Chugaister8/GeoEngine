/**
 * GeoEngine — Terrain Tile & Quadtree Tests
 */

import { describe, it, expect, beforeEach, vi } from "vitest"
import * as THREE from "three"

import { TerrainTile, buildHeightmapGeometry } from "../../packages/core-js/src/terrain/TerrainTile"
import {
  QuadtreeNode,
  QuadtreeLOD,
  DEFAULT_LOD_CONFIGS,
} from "../../packages/core-js/src/terrain/Quadtree"
import { tileToLatLonBBox } from "../../packages/core-js/src/geo/Coords"
import type { TileXYZ } from "../../packages/shared-types/src/geo"


// ================================================================
// TERRAIN TILE
// ================================================================

describe("TerrainTile", () => {

  const TILE: TileXYZ = { x: 0, y: 0, z: 5 }

  it("initial state is pending", () => {
    const tile = new TerrainTile(TILE, 0)
    expect(tile.state).toBe("pending")
    expect(tile.isReady).toBe(false)
  })

  it("tileKey is z/x/y format", () => {
    const tile = new TerrainTile({ x: 3, y: 7, z: 10 }, 0)
    expect(tile.tileKey).toBe("10/3/7")
  })

  it("bbox matches tile coordinates", () => {
    const tile     = new TerrainTile(TILE, 0)
    const expected = tileToLatLonBBox(TILE)
    expect(tile.bbox.west).toBeCloseTo(expected.west, 3)
    expect(tile.bbox.east).toBeCloseTo(expected.east, 3)
  })

  it("buildFromHeightmap creates ready mesh", () => {
    const tile     = new TerrainTile(TILE, 0)
    const w = 16, h = 16
    const heightmap = new Float32Array(w * h).fill(500)

    const material = new THREE.MeshStandardMaterial()
    tile.buildFromHeightmap(heightmap, w, h, 48.0, 23.0, material)

    expect(tile.state).toBe("ready")
    expect(tile.isReady).toBe(true)
    expect(tile.mesh).not.toBeNull()
    expect(tile.vertexCount).toBe(w * h)
    expect(tile.triangleCount).toBe((w-1) * (h-1) * 2)
  })

  it("dispose sets state to disposed", () => {
    const tile     = new TerrainTile(TILE, 0)
    const w = 4, h = 4
    const heightmap = new Float32Array(w * h).fill(100)
    const material  = new THREE.MeshStandardMaterial()

    tile.buildFromHeightmap(heightmap, w, h, 48.0, 23.0, material)
    tile.dispose()

    expect(tile.state).toBe("disposed")
    expect(tile.mesh).toBeNull()
  })

  it("touch updates lastUsed timestamp", async () => {
    const tile = new TerrainTile(TILE, 0)
    const before = tile.lastUsed

    await new Promise(r => setTimeout(r, 5))
    tile.touch()

    expect(tile.lastUsed).toBeGreaterThan(before)
  })
})


// ================================================================
// HEIGHTMAP GEOMETRY
// ================================================================

describe("buildHeightmapGeometry", () => {

  const BBOX = { west: 23.0, south: 48.0, east: 23.1, north: 48.1 }

  it("creates BufferGeometry with correct attribute count", () => {
    const w = 8, h = 8
    const hm  = new Float32Array(w * h).fill(100)
    const geo = buildHeightmapGeometry(hm, w, h, BBOX, 48.05, 23.05)

    expect(geo).toBeInstanceOf(THREE.BufferGeometry)
    expect(geo.attributes["position"]).toBeDefined()
    expect(geo.attributes["uv"]).toBeDefined()
    expect(geo.index).not.toBeNull()
  })

  it("vertex count = w × h", () => {
    const w = 6, h = 4
    const hm  = new Float32Array(w * h).fill(0)
    const geo = buildHeightmapGeometry(hm, w, h, BBOX, 48.05, 23.05)

    const positions = geo.attributes["position"] as THREE.BufferAttribute
    expect(positions.count).toBe(w * h)
  })

  it("index count = (w-1)×(h-1)×6", () => {
    const w = 5, h = 5
    const hm  = new Float32Array(w * h).fill(0)
    const geo = buildHeightmapGeometry(hm, w, h, BBOX, 48.05, 23.05)

    expect(geo.index!.count).toBe((w-1) * (h-1) * 6)
  })

  it("Y values correspond to heightmap values", () => {
    const w = 4, h = 4
    const hm = new Float32Array(w * h)

    // Перший піксель = 1000м, решта = 0
    hm[0] = 1000

    const geo      = buildHeightmapGeometry(hm, w, h, BBOX, 48.0, 23.0)
    const positions = geo.attributes["position"] as THREE.BufferAttribute

    // Y (index 1) першої вершини ≈ 1000
    const y0 = positions.getY(0)
    expect(y0).toBeCloseTo(1000, -1)   // ±10м
  })

  it("UV values in [0, 1]", () => {
    const w = 4, h = 4
    const hm  = new Float32Array(w * h).fill(500)
    const geo = buildHeightmapGeometry(hm, w, h, BBOX, 48.0, 23.0)
    const uvs = geo.attributes["uv"] as THREE.BufferAttribute

    for (let i = 0; i < uvs.count; i++) {
      expect(uvs.getX(i)).toBeGreaterThanOrEqual(0)
      expect(uvs.getX(i)).toBeLessThanOrEqual(1)
      expect(uvs.getY(i)).toBeGreaterThanOrEqual(0)
      expect(uvs.getY(i)).toBeLessThanOrEqual(1)
    }
  })
})


// ================================================================
// QUADTREE NODE
// ================================================================

describe("QuadtreeNode", () => {

  it("leaf node by default", () => {
    const node = new QuadtreeNode({ x: 0, y: 0, z: 5 })
    expect(node.isLeaf).toBe(true)
    expect(node.children).toBeNull()
  })

  it("split creates 4 children", () => {
    const node = new QuadtreeNode({ x: 0, y: 0, z: 5 })
    node.split()

    expect(node.isLeaf).toBe(false)
    expect(node.children).not.toBeNull()
    expect(node.children!.length).toBe(4)
  })

  it("children have correct zoom level", () => {
    const node = new QuadtreeNode({ x: 2, y: 3, z: 5 })
    node.split()

    for (const child of node.children!) {
      expect(child.tile.z).toBe(6)
    }
  })

  it("children have correct coordinates", () => {
    const node = new QuadtreeNode({ x: 2, y: 3, z: 5 })
    node.split()

    const [tl, tr, bl, br] = node.children!
    expect(tl.tile).toEqual({ x: 4, y: 6, z: 6 })
    expect(tr.tile).toEqual({ x: 5, y: 6, z: 6 })
    expect(bl.tile).toEqual({ x: 4, y: 7, z: 6 })
    expect(br.tile).toEqual({ x: 5, y: 7, z: 6 })
  })

  it("merge collapses children back", () => {
    const node = new QuadtreeNode({ x: 0, y: 0, z: 3 })
    node.split()
    expect(node.isLeaf).toBe(false)

    node.merge()
    expect(node.isLeaf).toBe(true)
    expect(node.children).toBeNull()
  })

  it("collectLeaves returns all leaf nodes", () => {
    const node = new QuadtreeNode({ x: 0, y: 0, z: 2 })
    node.split()
    node.children![0]!.split()  // Split one child

    const leaves = node.collectLeaves()
    // 1 split child = 4 leaves, 3 other children = 3 leaves → 7 total
    expect(leaves.length).toBe(7)
  })

  it("culled node not in leaves", () => {
    const node = new QuadtreeNode({ x: 0, y: 0, z: 2 })
    node.state = "culled"

    const leaves = node.collectLeaves()
    expect(leaves.length).toBe(0)
  })

  it("tileKey format is z/x/y", () => {
    const node = new QuadtreeNode({ x: 3, y: 5, z: 7 })
    expect(node.tileKey).toBe("7/3/5")
  })

  it("center is in bbox", () => {
    const tile = { x: 10, y: 12, z: 8 }
    const node = new QuadtreeNode(tile)
    const bbox = tileToLatLonBBox(tile)

    expect(node.centerLat).toBeGreaterThanOrEqual(bbox.south)
    expect(node.centerLat).toBeLessThanOrEqual(bbox.north)
    expect(node.centerLon).toBeGreaterThanOrEqual(bbox.west)
    expect(node.centerLon).toBeLessThanOrEqual(bbox.east)
  })
})


// ================================================================
// QUADTREE LOD
// ================================================================

describe("QuadtreeLOD", () => {

  it("creates with default config", () => {
    const lod = new QuadtreeLOD({ rootZoom: 3, maxZoom: 10 })
    expect(lod).toBeDefined()
  })

  it("getStats returns valid stats", () => {
    const lod   = new QuadtreeLOD({ rootZoom: 2, maxZoom: 8 })
    const stats = lod.getStats()

    expect(stats.total).toBeGreaterThan(0)
    expect(stats.leaves).toBeGreaterThan(0)
    expect(stats.maxDepth).toBeGreaterThanOrEqual(0)
    expect(stats.leaves).toBeLessThanOrEqual(stats.total)
  })

  it("getVisibleTiles returns array", () => {
    const lod     = new QuadtreeLOD({ rootZoom: 2, maxZoom: 8 })
    const frustum = new THREE.Frustum()

    // Встановлюємо frustum що охоплює все
    frustum.setFromProjectionMatrix(
      new THREE.Matrix4().makePerspective(-1, 1, 1, -1, 0.1, 100000)
    )

    const tiles = lod.getVisibleTiles(48.0, 23.0, 5000, frustum)
    expect(Array.isArray(tiles)).toBe(true)
  })

  it("closer camera gets more detailed tiles", () => {
    const lod = new QuadtreeLOD({
      rootZoom: 4,
      maxZoom:  14,
    })

    const frustum = new THREE.Frustum()
    frustum.setFromProjectionMatrix(
      new THREE.Matrix4().makePerspective(-1, 1, 1, -1, 0.1, 10_000_000)
    )

    // Близька камера
    const close = lod.getVisibleTiles(48.0, 23.0, 500,     frustum)
    // Далека камера
    const far   = lod.getVisibleTiles(48.0, 23.0, 500_000, frustum)

    // Близька камера має запитати деталізованіші тайли (більший zoom)
    const closeMaxZ = Math.max(...close.map(t => t.tile.z))
    const farMaxZ   = Math.max(...far.map(t => t.tile.z))

    expect(closeMaxZ).toBeGreaterThanOrEqual(farMaxZ)
  })

  it("tiles are sorted by distance (closest first)", () => {
    const lod     = new QuadtreeLOD({ rootZoom: 3, maxZoom: 10 })
    const frustum = new THREE.Frustum()
    frustum.setFromProjectionMatrix(
      new THREE.Matrix4().makePerspective(-1, 1, 1, -1, 0.1, 10_000_000)
    )

    const tiles = lod.getVisibleTiles(48.0, 23.0, 5000, frustum)

    for (let i = 0; i < tiles.length - 1; i++) {
      expect(tiles[i]!.distanceM).toBeLessThanOrEqual(tiles[i+1]!.distanceM)
    }
  })
})
