/**
 * GeoEngine — TerrainTile
 * Один тайл терейну на JS стороні.
 *
 * Отримує дані від Python сервера через WebSocket,
 * розпаковує буфери і готує Three.js BufferGeometry.
 *
 * Lifecycle:
 *   PENDING → LOADING → READY → DISPOSED
 */

import * as THREE from 'three'
import type { BBox, TileXYZ, LODLevel } from '../../shared-types/src/geo'
import { tileToLatLonBBox, tileResolutionM } from '../geo/Coords'

// ----------------------------------------------------------------
// СТАН ТАЙЛУ
// ----------------------------------------------------------------

export type TileState = 'pending' | 'loading' | 'ready' | 'error' | 'disposed'

// ----------------------------------------------------------------
// TERRAIN TILE
// ----------------------------------------------------------------

export class TerrainTile {
  /**
   * Один тайл терейну.
   *
   * Містить:
   * - Three.js Mesh (геометрія + матеріал)
   * - LOD рівень
   * - Метадані (bbox, висоти, resolution)
   * - Стан завантаження
   */

  readonly tileKey:   string     // "z/x/y"
  readonly tile:      TileXYZ
  readonly bbox:      BBox
  readonly lodLevel:  LODLevel

  private _state:     TileState = 'pending'
  private _mesh:      THREE.Mesh | null = null
  private _geometry:  THREE.BufferGeometry | null = null
  private _loadedAt:  number = 0
  private _lastUsed:  number = performance.now()

  // Статистика
  vertexCount:   number = 0
  triangleCount: number = 0
  memoryBytes:   number = 0

  // Висотні межі (для frustum culling)
  minElevation: number = 0
  maxElevation: number = 0

  constructor(tile: TileXYZ, lodLevel: LODLevel = 0) {
    this.tile     = tile
    this.tileKey  = `${tile.z}/${tile.x}/${tile.y}`
    this.bbox     = tileToLatLonBBox(tile)
    this.lodLevel = lodLevel
  }

  // ---- Getters ----

  get state(): TileState { return this._state }

  get isReady(): boolean { return this._state === 'ready' }

  get mesh(): THREE.Mesh | null { return this._mesh }

  get resolutionM(): number { return tileResolutionM(this.tile) }

  // ---- Побудова геометрії з даних сервера ----

  /**
   * Побудувати Three.js Mesh з даних отриманих від Python сервера.
   *
   * @param serverData — відповідь від ws/handler.py (to_dict())
   * @param material   — матеріал для меша (PBR terrain material)
   */
  buildFromServerData(
    serverData: TerrainMeshServerData,
    material:   THREE.Material,
  ): void {
    if (this._state === 'disposed') return

    this._state = 'loading'

    try {
      const geometry = this._decodeGeometry(serverData)
      this._mesh     = new THREE.Mesh(geometry, material)

      // Налаштування меша
      this._mesh.name           = `terrain_${this.tileKey}`
      this._mesh.receiveShadow  = true
      this._mesh.castShadow     = false

      // Frustum culling bounding box
      geometry.computeBoundingBox()
      geometry.computeBoundingSphere()

      this._geometry    = geometry
      this.vertexCount  = serverData.vertex_count
      this.triangleCount = serverData.triangle_count
      this.memoryBytes  = this._estimateMemory()
      this._state       = 'ready'
      this._loadedAt    = performance.now()

    } catch (err) {
      this._state = 'error'
      console.error(`TerrainTile ${this.tileKey}: build error`, err)
    }
  }

  /**
   * Побудувати меш безпосередньо з heightmap Float32Array.
   * Використовується коли дані вже на клієнті (Terrarium tiles).
   *
   * @param heightmap  — Float32Array висот (width×height, row-major)
   * @param width      — кількість стовпців
   * @param height     — кількість рядків
   * @param originLat  — широта origin (0,0,0)
   * @param originLon  — довгота origin
   * @param material   — матеріал
   */
  buildFromHeightmap(
    heightmap: Float32Array,
    width:     number,
    height:    number,
    originLat: number,
    originLon: number,
    material:  THREE.Material,
  ): void {
    if (this._state === 'disposed') return
    this._state = 'loading'

    try {
      const geometry = buildHeightmapGeometry(
        heightmap, width, height,
        this.bbox, originLat, originLon,
      )

      this._mesh = new THREE.Mesh(geometry, material)
      this._mesh.name = `terrain_${this.tileKey}`
      this._mesh.receiveShadow = true
      this._mesh.castShadow    = false

      geometry.computeBoundingBox()
      geometry.computeBoundingSphere()

      this._geometry     = geometry
      this.vertexCount   = width * height
      this.triangleCount = (width - 1) * (height - 1) * 2
      this.memoryBytes   = this._estimateMemory()
      this._state        = 'ready'
      this._loadedAt     = performance.now()

    } catch (err) {
      this._state = 'error'
      console.error(`TerrainTile ${this.tileKey}: heightmap build error`, err)
    }
  }

  /**
   * Оновити LOD морфінг (geomorphing між рівнями).
   * Викликається кожен кадр для плавного переходу LOD.
   *
   * @param morphFactor — 0.0 (поточний LOD) .. 1.0 (наступний LOD)
   */
  updateMorphFactor(morphFactor: number): void {
    if (!this._mesh) return
    const mat = this._mesh.material as THREE.ShaderMaterial
    if (mat.uniforms?.['morphFactor']) {
      mat.uniforms['morphFactor'].value = Math.max(0, Math.min(1, morphFactor))
    }
  }

  touch(): void {
    this._lastUsed = performance.now()
  }

  get lastUsed(): number { return this._lastUsed }

  get ageMs(): number { return performance.now() - this._loadedAt }

  // ---- Dispose ----

  dispose(): void {
    if (this._state === 'disposed') return

    this._geometry?.dispose()
    // Матеріал не диспоузимо — він shared між тайлами
    this._mesh     = null
    this._geometry = null
    this._state    = 'disposed'
  }

  // ---- Приватні методи ----

  private _decodeGeometry(data: TerrainMeshServerData): THREE.BufferGeometry {
    const geo = new THREE.BufferGeometry()

    const vertices = _decodeBase64Float32(data.buffers.vertices)
    const indices  = _decodeBase64Uint32(data.buffers.indices)
    const uvs      = _decodeBase64Float32(data.buffers.uvs)
    const normals  = _decodeBase64Float32(data.buffers.normals)

    geo.setAttribute('position', new THREE.BufferAttribute(vertices, 3))
    geo.setAttribute('uv',       new THREE.BufferAttribute(uvs, 2))
    geo.setAttribute('normal',   new THREE.BufferAttribute(normals, 3))
    geo.setIndex(new THREE.BufferAttribute(indices, 1))

    // Висотні межі для LOD culling
    const posArr = vertices
    let minY = Infinity
    let maxY = -Infinity
    for (let i = 1; i < posArr.length; i += 3) {
      const y = posArr[i]!
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
    this.minElevation = minY
    this.maxElevation = maxY

    return geo
  }

  private _estimateMemory(): number {
    if (!this._geometry) return 0
    let bytes = 0
    for (const attr of Object.values(this._geometry.attributes)) {
      bytes += (attr as THREE.BufferAttribute).array.byteLength
    }
    if (this._geometry.index) {
      bytes += this._geometry.index.array.byteLength
    }
    return bytes
  }

  toString(): string {
    return `TerrainTile(${this.tileKey}, lod=${this.lodLevel}, state=${this._state})`
  }
}

// ----------------------------------------------------------------
// ТИПИ СЕРВЕРНИХ ДАНИХ
// ----------------------------------------------------------------

/** Формат відповіді від Python TerrainMesh.to_dict() */
export interface TerrainMeshServerData {
  readonly type:           'terrain_mesh'
  readonly lod_level:      number
  readonly vertex_count:   number
  readonly triangle_count: number
  readonly bbox:           [number, number, number, number]  // [W, S, E, N]
  readonly origin: {
    readonly lat: number
    readonly lon: number
    readonly alt: number
  }
  readonly buffers: {
    readonly vertices: string   // base64(Float32Array)
    readonly indices:  string   // base64(Uint32Array)
    readonly uvs:      string   // base64(Float32Array)
    readonly normals:  string   // base64(Float32Array)
  }
}

// ----------------------------------------------------------------
// HEIGHTMAP → THREE.JS GEOMETRY
// ----------------------------------------------------------------

/**
 * Побудувати BufferGeometry з heightmap безпосередньо на клієнті.
 * Використовується для Terrarium tiles (PNG → висота).
 */
export function buildHeightmapGeometry(
  heightmap: Float32Array,
  w:         number,
  h:         number,
  bbox:      BBox,
  originLat: number,
  originLon: number,
): THREE.BufferGeometry {
  const N = w * h

  const positions = new Float32Array(N * 3)
  const uvs       = new Float32Array(N * 2)

  const R      = 6_378_137.0
  const cosLat = Math.cos(originLat * Math.PI / 180)

  // Сітка координат
  for (let row = 0; row < h; row++) {
    for (let col = 0; col < w; col++) {
      const idx = row * w + col

      const lat = bbox.north - (row / (h - 1)) * (bbox.north - bbox.south)
      const lon = bbox.west  + (col / (w - 1)) * (bbox.east  - bbox.west)
      const alt = heightmap[idx] ?? 0

      // ENU → Three.js (Y вгору, Z = -North)
      const east  = (lon - originLon) * (Math.PI / 180) * cosLat * R
      const north = (lat - originLat) * (Math.PI / 180) * R
      const up    = alt - 0  // originAlt = 0

      positions[idx * 3 + 0] =  east
      positions[idx * 3 + 1] =  up
      positions[idx * 3 + 2] = -north

      uvs[idx * 2 + 0] = col / (w - 1)
      uvs[idx * 2 + 1] = row / (h - 1)
    }
  }

  // Індекси (два трикутники на квадрат)
  const numQuads  = (w - 1) * (h - 1)
  const indexArr  = new Uint32Array(numQuads * 6)
  let   idxOffset = 0

  for (let row = 0; row < h - 1; row++) {
    for (let col = 0; col < w - 1; col++) {
      const tl = row * w + col
      const tr = tl + 1
      const bl = tl + w
      const br = bl + 1

      // CCW (counter-clockwise)
      indexArr[idxOffset++] = tl
      indexArr[idxOffset++] = bl
      indexArr[idxOffset++] = tr
      indexArr[idxOffset++] = tr
      indexArr[idxOffset++] = bl
      indexArr[idxOffset++] = br
    }
  }

  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geo.setAttribute('uv',       new THREE.BufferAttribute(uvs, 2))
  geo.setIndex(new THREE.BufferAttribute(indexArr, 1))
  geo.computeVertexNormals()

  return geo
}

// ----------------------------------------------------------------
// ДЕКОДУВАННЯ BASE64 БУФЕРІВ
// ----------------------------------------------------------------

function _decodeBase64Float32(b64: string): Float32Array {
  const binary = atob(b64)
  const buffer = new ArrayBuffer(binary.length)
  const view   = new Uint8Array(buffer)
  for (let i = 0; i < binary.length; i++) {
    view[i] = binary.charCodeAt(i)
  }
  return new Float32Array(buffer)
}

function _decodeBase64Uint32(b64: string): Uint32Array {
  const binary = atob(b64)
  const buffer = new ArrayBuffer(binary.length)
  const view   = new Uint8Array(buffer)
  for (let i = 0; i < binary.length; i++) {
    view[i] = binary.charCodeAt(i)
  }
  return new Uint32Array(buffer)
}
