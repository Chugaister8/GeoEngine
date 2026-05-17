/**
 * GeoEngine — Coordinate Tests (JavaScript/TypeScript)
 * Vitest unit tests для geo/Coords.ts
 */

import { describe, it, expect } from "vitest"
import {
  llhToECEF,
  ecefToLLH,
  llhToENU,
  enuToThreeJS,
  llhToWorld,
  llhToWebMercator,
  webMercatorToLLH,
  latLonToTile,
  tileToLatLonBBox,
  bboxToTiles,
  tileResolutionM,
  haversineDistance,
  bearing,
  WGS84,
  DEG2RAD,
} from "../../packages/core-js/src/geo/Coords"


// ================================================================
// LLH ↔ ECEF
// ================================================================

describe("llhToECEF", () => {

  it("equator prime meridian → (R, 0, 0)", () => {
    const ecef = llhToECEF(0, 0, 0)
    expect(ecef.x).toBeCloseTo(WGS84.A, -1)  // ±10м
    expect(ecef.y).toBeCloseTo(0, 0)
    expect(ecef.z).toBeCloseTo(0, 0)
  })

  it("north pole → (0, 0, b)", () => {
    const ecef = llhToECEF(90, 0, 0)
    expect(ecef.x).toBeCloseTo(0, -1)
    expect(ecef.y).toBeCloseTo(0, -1)
    expect(ecef.z).toBeCloseTo(WGS84.B, -2)
  })

  it("round trip: LLH → ECEF → LLH", () => {
    const lat = 50.45, lon = 30.52, alt = 200
    const ecef = llhToECEF(lat, lon, alt)
    const back = ecefToLLH(ecef.x, ecef.y, ecef.z)

    expect(back.lat).toBeCloseTo(lat, 6)
    expect(back.lon).toBeCloseTo(lon, 6)
    expect(back.alt).toBeCloseTo(alt, 2)
  })

  it("round trip: Hoverla (highest point Ukraine)", () => {
    const lat = 48.16, lon = 24.50, alt = 2061
    const ecef = llhToECEF(lat, lon, alt)
    const back = ecefToLLH(ecef.x, ecef.y, ecef.z)

    expect(back.lat).toBeCloseTo(lat, 5)
    expect(back.lon).toBeCloseTo(lon, 5)
    expect(back.alt).toBeCloseTo(alt, 1)
  })
})


// ================================================================
// LLH ↔ ENU
// ================================================================

describe("llhToENU", () => {

  const ORIGIN = { lat: 48.0, lon: 23.0, alt: 0 }

  it("origin point → ENU (0, 0, 0)", () => {
    const enu = llhToENU(ORIGIN.lat, ORIGIN.lon, ORIGIN.alt,
                          ORIGIN.lat, ORIGIN.lon, ORIGIN.alt)
    expect(enu.east).toBeCloseTo(0, 1)
    expect(enu.north).toBeCloseTo(0, 1)
    expect(enu.up).toBeCloseTo(0, 1)
  })

  it("point east of origin → positive East", () => {
    const enu = llhToENU(48.0, 23.01, 0, ORIGIN.lat, ORIGIN.lon)
    expect(enu.east).toBeGreaterThan(0)
    expect(Math.abs(enu.north)).toBeLessThan(1)
  })

  it("point north of origin → positive North", () => {
    const enu = llhToENU(48.01, 23.0, 0, ORIGIN.lat, ORIGIN.lon)
    expect(enu.north).toBeGreaterThan(0)
    expect(Math.abs(enu.east)).toBeLessThan(1)
  })

  it("point above origin → positive Up", () => {
    const enu = llhToENU(48.0, 23.0, 1000, ORIGIN.lat, ORIGIN.lon)
    expect(enu.up).toBeCloseTo(1000, 0)
  })
})

describe("enuToThreeJS", () => {

  it("ENU → Three.js axes (X=East, Y=Up, Z=-North)", () => {
    const enu = { east: 100, north: 200, up: 300 }
    const v3  = enuToThreeJS(enu)
    expect(v3.x).toBe(100)   // East → X
    expect(v3.y).toBe(300)   // Up → Y
    expect(v3.z).toBe(-200)  // -North → Z
  })
})


// ================================================================
// WebMercator
// ================================================================

describe("WebMercator", () => {

  it("(0, 0) → WebMercator (0, 0)", () => {
    const wm = llhToWebMercator(0, 0)
    expect(wm.x).toBeCloseTo(0, 0)
    expect(wm.y).toBeCloseTo(0, 0)
  })

  it("round trip lat/lon → WebMercator → lat/lon", () => {
    const lat = 48.0, lon = 23.0
    const wm  = llhToWebMercator(lat, lon)
    const back = webMercatorToLLH(wm.x, wm.y)

    expect(back.lat).toBeCloseTo(lat, 5)
    expect(back.lon).toBeCloseTo(lon, 5)
  })

  it("returns finite numbers for extreme latitudes", () => {
    const wm = llhToWebMercator(85.0, 0)
    expect(isFinite(wm.x)).toBe(true)
    expect(isFinite(wm.y)).toBe(true)
  })
})


// ================================================================
// TILE ADDRESSING
// ================================================================

describe("latLonToTile", () => {

  it("any point at zoom 0 → (0,0,0)", () => {
    const t1 = latLonToTile(48.0, 23.0, 0)
    const t2 = latLonToTile(-45.0, 120.0, 0)
    expect(t1).toEqual({ x: 0, y: 0, z: 0 })
    expect(t2).toEqual({ x: 0, y: 0, z: 0 })
  })

  it("tile bbox contains original point", () => {
    const lat = 48.15, lon = 24.50, zoom = 12
    const tile = latLonToTile(lat, lon, zoom)
    const bbox = tileToLatLonBBox(tile)

    expect(lat).toBeGreaterThanOrEqual(bbox.south)
    expect(lat).toBeLessThanOrEqual(bbox.north)
    expect(lon).toBeGreaterThanOrEqual(bbox.west)
    expect(lon).toBeLessThanOrEqual(bbox.east)
  })

  it("higher zoom → smaller bbox", () => {
    const lat = 48.0, lon = 23.0
    const t5  = latLonToTile(lat, lon, 5)
    const t10 = latLonToTile(lat, lon, 10)

    const b5  = tileToLatLonBBox(t5)
    const b10 = tileToLatLonBBox(t10)

    const w5  = b5.east - b5.west
    const w10 = b10.east - b10.west
    expect(w10).toBeLessThan(w5)
  })
})

describe("tileToLatLonBBox", () => {

  it("zoom 0 → entire world", () => {
    const bbox = tileToLatLonBBox({ x: 0, y: 0, z: 0 })
    expect(bbox.west).toBeCloseTo(-180, 0)
    expect(bbox.east).toBeCloseTo(180, 0)
    expect(bbox.north).toBeGreaterThan(84)
    expect(bbox.south).toBeLessThan(-84)
  })

  it("zoom 1 has 4 tiles covering world", () => {
    const tiles = [
      { x: 0, y: 0, z: 1 },
      { x: 1, y: 0, z: 1 },
      { x: 0, y: 1, z: 1 },
      { x: 1, y: 1, z: 1 },
    ]
    const bboxes = tiles.map(tileToLatLonBBox)
    const west   = Math.min(...bboxes.map(b => b.west))
    const east   = Math.max(...bboxes.map(b => b.east))
    expect(west).toBeCloseTo(-180, 0)
    expect(east).toBeCloseTo(180, 0)
  })
})

describe("bboxToTiles", () => {

  it("small bbox → at least 1 tile", () => {
    const bbox  = { west: 23.0, south: 48.0, east: 23.5, north: 48.5 }
    const tiles = bboxToTiles(bbox, 8)
    expect(tiles.length).toBeGreaterThan(0)
  })

  it("all tiles cover the bbox", () => {
    const bbox  = { west: 23.0, south: 48.0, east: 24.0, north: 49.0 }
    const zoom  = 8
    const tiles = bboxToTiles(bbox, zoom)
    const bboxes = tiles.map(tileToLatLonBBox)

    const minWest  = Math.min(...bboxes.map(b => b.west))
    const minSouth = Math.min(...bboxes.map(b => b.south))
    const maxEast  = Math.max(...bboxes.map(b => b.east))
    const maxNorth = Math.max(...bboxes.map(b => b.north))

    expect(minWest).toBeLessThanOrEqual(bbox.west)
    expect(minSouth).toBeLessThanOrEqual(bbox.south)
    expect(maxEast).toBeGreaterThanOrEqual(bbox.east)
    expect(maxNorth).toBeGreaterThanOrEqual(bbox.north)
  })
})


// ================================================================
// RESOLUTION
// ================================================================

describe("tileResolutionM", () => {

  it("zoom 0 ≈ 156km/px at equator", () => {
    const res = tileResolutionM({ x: 0, y: 0, z: 0 })
    expect(res).toBeCloseTo(156_543, -3)   // ±1000м
  })

  it("resolution decreases with zoom", () => {
    const r5  = tileResolutionM({ x: 0, y: 0, z: 5 })
    const r10 = tileResolutionM({ x: 0, y: 0, z: 10 })
    expect(r10).toBeLessThan(r5)
  })
})


// ================================================================
// ВІДСТАНІ
// ================================================================

describe("haversineDistance", () => {

  it("same point → 0", () => {
    expect(haversineDistance(48, 23, 48, 23)).toBeCloseTo(0, 0)
  })

  it("1° on equator ≈ 111320m", () => {
    const dist = haversineDistance(0, 0, 0, 1)
    expect(dist).toBeCloseTo(111_320, -2)   // ±100м
  })

  it("Kyiv to Lviv ≈ 470km", () => {
    const dist = haversineDistance(50.45, 30.52, 49.84, 24.02)
    expect(dist).toBeCloseTo(470_000, -4)   // ±10км
  })
})

describe("bearing", () => {

  it("east direction → 90°", () => {
    const b = bearing(48, 23, 48, 24)
    expect(b).toBeCloseTo(90, 0)
  })

  it("north direction → 0° (or 360°)", () => {
    const b = bearing(48, 23, 49, 23)
    expect(b % 360).toBeCloseTo(0, 0)
  })

  it("south direction → 180°", () => {
    const b = bearing(48, 23, 47, 23)
    expect(b).toBeCloseTo(180, 0)
  })

  it("result always in [0, 360)", () => {
    const pairs = [
      [0, 0, 1, 1],
      [0, 0, -1, -1],
      [45, 90, -45, -90],
    ] as const

    for (const [lat1, lon1, lat2, lon2] of pairs) {
      const b = bearing(lat1, lon1, lat2, lon2)
      expect(b).toBeGreaterThanOrEqual(0)
      expect(b).toBeLessThan(360)
    }
  })
})
