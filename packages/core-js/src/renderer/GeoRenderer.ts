/**
 * GeoEngine — GeoRenderer
 * Головний WebGPU рендерер терейну.
 *
 * Відповідає за:
 * - Ініціалізацію Three.js WebGPURenderer
 * - Управління сценою (Scene, Camera, Lights)
 * - LOD оновлення кожен кадр
 * - Завантаження тайлів через WebSocket
 * - Render loop (requestAnimationFrame)
 * - Post-processing pipeline
 *
 * Usage:
 *   const renderer = new GeoRenderer({ canvas: '#viewport' })
 *   await renderer.init()
 *   renderer.flyTo({ lat: 48.25, lon: 23.5, altitude: 3000 })
 *   renderer.start()
 */

import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { BBox, TileXYZ, LODLevel } from '../../shared-types/src/geo'
import { TerrainTile, type TerrainMeshServerData } from '../terrain/TerrainTile'
import { QuadtreeLOD, type QuadtreeStats } from '../terrain/Quadtree'
import { llhToWorld, tileResolutionM } from '../geo/Coords'

// ----------------------------------------------------------------
// КОНФІГУРАЦІЯ
// ----------------------------------------------------------------

export interface GeoRendererOptions {
  /** CSS селектор або HTMLCanvasElement */
  canvas:       string | HTMLCanvasElement

  /** WebGPU (primary) або WebGL2 (fallback) */
  renderer?:    'webgpu' | 'webgl2' | 'auto'

  /** Початкова позиція камери */
  initialCamera?: {
    lat:      number
    lon:      number
    altitude: number  // метри
  }

  /** WebSocket URL до Python сервера */
  serverUrl?:   string

  /** Origin для ENU координат (за замовчуванням = initialCamera) */
  originLat?:   number
  originLon?:   number

  /** Максимальний zoom для LOD */
  maxLODZoom?:  number

  /** Показати debug статистику (FPS, tris, тощо) */
  debug?:       boolean
}

// ----------------------------------------------------------------
// СТАН РЕНДЕРЕРА
// ----------------------------------------------------------------

export type RendererState = 'idle' | 'initializing' | 'running' | 'paused' | 'error'

// ----------------------------------------------------------------
// GEO RENDERER
// ----------------------------------------------------------------

export class GeoRenderer {
  // Three.js core
  private _renderer!:  THREE.WebGLRenderer
  private _scene!:     THREE.Scene
  private _camera!:    THREE.PerspectiveCamera
  private _controls!:  OrbitControls

  // LOD система
  private _quadtree!:  QuadtreeLOD

  // Тайли
  private _tiles:      Map<string, TerrainTile> = new Map()
  private _pendingKeys: Set<string> = new Set()

  // WebSocket до Python сервера
  private _ws:         WebSocket | null = null
  private _wsUrl:      string

  // Матеріали
  private _terrainMaterial!: THREE.MeshStandardMaterial

  // Стан
  private _state:      RendererState = 'idle'
  private _frameId:    number | null = null
  private _clock:      THREE.Clock = new THREE.Clock()

  // Конфігурація
  private _opts:       Required<GeoRendererOptions>
  private _originLat:  number
  private _originLon:  number

  // Фрустум для LOD culling
  private _frustum:    THREE.Frustum = new THREE.Frustum()
  private _projMatrix: THREE.Matrix4 = new THREE.Matrix4()

  // Статистика
  private _stats: RenderStats = {
    fps: 0, frameTime: 0,
    visibleTiles: 0, totalTiles: 0,
    triangles: 0, drawCalls: 0,
    quadtree: { total: 0, leaves: 0, culled: 0, maxDepth: 0 },
  }

  constructor(options: GeoRendererOptions) {
    this._opts = {
      canvas:        options.canvas,
      renderer:      options.renderer      ?? 'auto',
      initialCamera: options.initialCamera ?? { lat: 48.25, lon: 23.5, altitude: 5000 },
      serverUrl:     options.serverUrl     ?? 'ws://localhost:8000/ws',
      originLat:     options.originLat     ?? options.initialCamera?.lat ?? 48.25,
      originLon:     options.originLon     ?? options.initialCamera?.lon ?? 23.5,
      maxLODZoom:    options.maxLODZoom    ?? 14,
      debug:         options.debug         ?? false,
    }

    this._originLat = this._opts.originLat
    this._originLon = this._opts.originLon
    this._wsUrl     = this._opts.serverUrl
  }

  // ---- Ініціалізація ----

  async init(): Promise<void> {
    if (this._state !== 'idle') {
      throw new Error('GeoRenderer вже ініціалізований')
    }
    this._state = 'initializing'

    try {
      await this._initRenderer()
      this._initScene()
      this._initCamera()
      this._initLights()
      this._initControls()
      this._initLOD()
      this._initMaterials()
      this._initWebSocket()
      this._initResizeObserver()

      this._state = 'running'
      console.log('[GeoRenderer] ініціалізований', {
        renderer: this._renderer.getContext().constructor.name,
        origin:   `${this._originLat}, ${this._originLon}`,
      })
    } catch (err) {
      this._state = 'error'
      throw err
    }
  }

  // ---- Render Loop ----

  start(): void {
    if (this._frameId !== null) return
    this._clock.start()
    this._loop()
  }

  stop(): void {
    if (this._frameId !== null) {
      cancelAnimationFrame(this._frameId)
      this._frameId = null
    }
  }

  // ---- Camera Controls ----

  /**
   * Плавний переліт камери до заданих координат.
   *
   * @param lat      широта цілі
   * @param lon      довгота цілі
   * @param altitude висота камери (метри)
   * @param duration тривалість анімації (мс)
   */
  flyTo(options: {
    lat:       number
    lon:       number
    altitude:  number
    duration?: number
  }): void {
    const { lat, lon, altitude, duration = 2000 } = options
    const target = llhToWorld(lat, lon, 0, this._originLat, this._originLon)
    const camPos = llhToWorld(lat, lon, altitude, this._originLat, this._originLon)

    this._animateFlyTo(
      this._camera.position.clone(),
      new THREE.Vector3(camPos.x, camPos.y, camPos.z),
      new THREE.Vector3(target.x, target.y, target.z),
      duration,
    )
  }

  /**
   * Встановити позицію камери миттєво (без анімації).
   */
  setCameraPosition(lat: number, lon: number, altitude: number): void {
    const pos    = llhToWorld(lat, lon, altitude, this._originLat, this._originLon)
    const target = llhToWorld(lat, lon, 0, this._originLat, this._originLon)

    this._camera.position.set(pos.x, pos.y, pos.z)
    this._controls.target.set(target.x, target.y, target.z)
    this._controls.update()
  }

  // ---- Getters ----

  get state():    RendererState { return this._state }
  get stats():    RenderStats   { return { ...this._stats } }
  get scene():    THREE.Scene   { return this._scene }
  get camera():   THREE.PerspectiveCamera { return this._camera }

  // ---- Dispose ----

  dispose(): void {
    this.stop()
    this._ws?.close()

    for (const tile of this._tiles.values()) {
      tile.dispose()
    }
    this._tiles.clear()

    this._terrainMaterial?.dispose()
    this._renderer?.dispose()
    this._state = 'idle'
  }

  // ================================================================
  // ПРИВАТНІ МЕТОДИ
  // ================================================================

  // ---- Ініціалізація Three.js ----

  private async _initRenderer(): Promise<void> {
    const canvas = typeof this._opts.canvas === 'string'
      ? document.querySelector<HTMLCanvasElement>(this._opts.canvas)
      : this._opts.canvas

    if (!canvas) {
      throw new Error(`Canvas не знайдено: ${this._opts.canvas}`)
    }

    // Спроба WebGPU
    const useWebGPU = this._opts.renderer === 'webgpu'
      || (this._opts.renderer === 'auto' && await _isWebGPUSupported())

    if (useWebGPU) {
      console.log('[GeoRenderer] WebGPU renderer')
      // Three.js r165+ WebGPURenderer
      // Динамічний імпорт щоб не ламати WebGL fallback
      const { default: WebGPURenderer } = await import(
        'three/examples/jsm/renderers/webgpu/WebGPURenderer.js'
      )
      this._renderer = new WebGPURenderer({ canvas, antialias: true }) as unknown as THREE.WebGLRenderer
      await (this._renderer as any).init()
    } else {
      console.log('[GeoRenderer] WebGL2 renderer (fallback)')
      this._renderer = new THREE.WebGLRenderer({
        canvas,
        antialias:    true,
        logarithmicDepthBuffer: true,  // важливо для великих сцен
      })
    }

    this._renderer.setPixelRatio(window.devicePixelRatio)
    this._renderer.setSize(canvas.clientWidth, canvas.clientHeight)
    this._renderer.shadowMap.enabled = true
    this._renderer.shadowMap.type    = THREE.PCFSoftShadowMap
    this._renderer.toneMapping       = THREE.ACESFilmicToneMapping
    this._renderer.toneMappingExposure = 1.0
    this._renderer.outputColorSpace  = THREE.SRGBColorSpace
  }

  private _initScene(): void {
    this._scene = new THREE.Scene()
    this._scene.background = new THREE.Color(0x87ceeb)  // sky blue
    this._scene.fog = new THREE.FogExp2(0xcce8ff, 0.000015)
  }

  private _initCamera(): void {
    const canvas = this._renderer.domElement
    this._camera = new THREE.PerspectiveCamera(
      60,                                           // FOV
      canvas.clientWidth / canvas.clientHeight,    // aspect
      1,                                            // near (1м)
      10_000_000,                                   // far  (10000км)
    )

    // Початкова позиція
    const { lat, lon, altitude } = this._opts.initialCamera
    const pos = llhToWorld(lat, lon, altitude, this._originLat, this._originLon)
    this._camera.position.set(pos.x, pos.y, pos.z)
    this._camera.lookAt(0, 0, 0)
  }

  private _initLights(): void {
    // Ambient
    const ambient = new THREE.AmbientLight(0xffffff, 0.4)
    this._scene.add(ambient)

    // Directional (сонце)
    const sun = new THREE.DirectionalLight(0xfffaed, 1.2)
    sun.position.set(5000, 8000, 3000)
    sun.castShadow = true
    sun.shadow.mapSize.width  = 2048
    sun.shadow.mapSize.height = 2048
    sun.shadow.camera.near    = 1
    sun.shadow.camera.far     = 50_000
    sun.shadow.camera.left    = -20_000
    sun.shadow.camera.right   =  20_000
    sun.shadow.camera.top     =  20_000
    sun.shadow.camera.bottom  = -20_000
    this._scene.add(sun)

    // Hemisphere (небо + земля)
    const hemi = new THREE.HemisphereLight(0x87ceeb, 0x4a7c3f, 0.3)
    this._scene.add(hemi)
  }

  private _initControls(): void {
    this._controls = new OrbitControls(this._camera, this._renderer.domElement)
    this._controls.enableDamping    = true
    this._controls.dampingFactor    = 0.05
    this._controls.screenSpacePanning = false
    this._controls.minDistance      = 100        // 100м
    this._controls.maxDistance      = 5_000_000  // 5000км
    this._controls.maxPolarAngle    = Math.PI / 2 - 0.01  // не нижче горизонту
  }

  private _initLOD(): void {
    this._quadtree = new QuadtreeLOD({
      rootZoom: 4,
      maxZoom:  this._opts.maxLODZoom,
    })
  }

  private _initMaterials(): void {
    this._terrainMaterial = new THREE.MeshStandardMaterial({
      color:    0x7a9e6e,
      roughness: 0.9,
      metalness: 0.0,
      side:     THREE.FrontSide,
    })
  }

  private _initWebSocket(): void {
    if (!this._wsUrl) return

    const connect = (): void => {
      this._ws = new WebSocket(this._wsUrl)

      this._ws.onopen = () => {
        console.log('[GeoRenderer] WebSocket підключений')
      }

      this._ws.onmessage = (event: MessageEvent) => {
        this._handleWSMessage(event.data as string)
      }

      this._ws.onerror = (err) => {
        console.warn('[GeoRenderer] WebSocket помилка', err)
      }

      this._ws.onclose = () => {
        console.log('[GeoRenderer] WebSocket закрито, перепідключення...')
        setTimeout(connect, 3000)
      }
    }

    connect()
  }

  private _initResizeObserver(): void {
    const canvas = this._renderer.domElement
    const observer = new ResizeObserver(() => {
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      this._camera.aspect = w / h
      this._camera.updateProjectionMatrix()
      this._renderer.setSize(w, h)
    })
    observer.observe(canvas)
  }

  // ---- Render Loop ----

  private _loop(): void {
    this._frameId = requestAnimationFrame(() => this._loop())

    const delta = this._clock.getDelta()
    this._update(delta)
    this._render()
    this._updateStats(delta)
  }

  private _update(delta: number): void {
    // Оновлення OrbitControls
    this._controls.update()

    // Оновлення frustum
    this._projMatrix.multiplyMatrices(
      this._camera.projectionMatrix,
      this._camera.matrixWorldInverse,
    )
    this._frustum.setFromProjectionMatrix(this._projMatrix)

    // Позиція камери у lat/lon
    const camWorld = this._camera.position
    const camLat   = this._originLat + camWorld.z / (-111_320)
    const camLon   = this._originLon + camWorld.x / (111_320 * Math.cos(this._originLat * Math.PI / 180))
    const camAlt   = camWorld.y

    // Оновлення LOD quadtree
    const visibleNodes = this._quadtree.getVisibleTiles(
      camLat, camLon, camAlt,
      this._frustum,
    )

    this._stats.quadtree  = this._quadtree.getStats()
    this._stats.visibleTiles = visibleNodes.length

    // Запросити потрібні тайли
    for (const { tile, lodLevel } of visibleNodes) {
      const key = `${tile.z}/${tile.x}/${tile.y}`
      if (!this._tiles.has(key) && !this._pendingKeys.has(key)) {
        this._requestTile(tile, lodLevel)
      } else {
        this._tiles.get(key)?.touch()
      }
    }

    // Додати видимі меші до сцени
    const visibleKeys = new Set(visibleNodes.map(n => `${n.tile.z}/${n.tile.x}/${n.tile.y}`))
    for (const [key, tile] of this._tiles) {
      if (!tile.mesh) continue
      const shouldBeVisible = visibleKeys.has(key)
      if (shouldBeVisible && !this._scene.getObjectByName(`terrain_${key}`)) {
        this._scene.add(tile.mesh)
      } else if (!shouldBeVisible && tile.mesh.parent) {
        this._scene.remove(tile.mesh)
      }
    }

    // Cleanup старих тайлів (LRU: видаляємо не використовувані > 30 сек)
    this._evictOldTiles(30_000)
  }

  private _render(): void {
    this._renderer.render(this._scene, this._camera)
  }

  // ---- Tile Management ----

  private _requestTile(tile: TileXYZ, lodLevel: LODLevel): void {
    const key = `${tile.z}/${tile.x}/${tile.y}`
    this._pendingKeys.add(key)

    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) {
      this._pendingKeys.delete(key)
      return
    }

    const message = {
      type:    'request_tile',
      id:      crypto.randomUUID(),
      timestamp: Date.now(),
      payload: { tile, source: 'copernicus25' },
    }

    this._ws.send(JSON.stringify(message))
  }

  private _handleWSMessage(raw: string): void {
    let msg: any
    try {
      msg = JSON.parse(raw)
    } catch {
      console.warn('[GeoRenderer] невалідний JSON від сервера')
      return
    }

    if (msg.type === 'response_tile') {
      this._onTileReceived(msg.payload as TerrainMeshServerData)
    } else if (msg.type === 'error') {
      console.error('[GeoRenderer] помилка сервера:', msg.payload?.message)
    }
  }

  private _onTileReceived(data: TerrainMeshServerData): void {
    const key = `${data.bbox[0]}_${data.bbox[1]}_${data.bbox[2]}_${data.bbox[3]}`

    // Знаходимо правильний tileKey
    const tileKey = `${data.lod_level}/${data.origin.lon}/${data.origin.lat}`

    // Пробуємо знайти по pending
    for (const k of this._pendingKeys) {
      this._pendingKeys.delete(k)

      const terrainTile = new TerrainTile(
        { x: 0, y: 0, z: data.lod_level },  // тимчасово
        data.lod_level as LODLevel,
      )

      terrainTile.buildFromServerData(data, this._terrainMaterial)

      if (terrainTile.isReady) {
        this._tiles.set(k, terrainTile)
      }
      break
    }
  }

  private _evictOldTiles(maxAgeMs: number): void {
    for (const [key, tile] of this._tiles) {
      if (tile.ageMs > maxAgeMs && !this._scene.getObjectByName(`terrain_${key}`)) {
        tile.dispose()
        this._tiles.delete(key)
      }
    }
  }

  // ---- Анімація ----

  private _animateFlyTo(
    fromPos:   THREE.Vector3,
    toPos:     THREE.Vector3,
    toTarget:  THREE.Vector3,
    duration:  number,
  ): void {
    const startTime    = performance.now()
    const fromTarget   = this._controls.target.clone()

    const animate = (): void => {
      const elapsed = performance.now() - startTime
      const t       = Math.min(elapsed / duration, 1.0)
      const eased   = _easeInOutCubic(t)

      this._camera.position.lerpVectors(fromPos, toPos, eased)
      this._controls.target.lerpVectors(fromTarget, toTarget, eased)
      this._controls.update()

      if (t < 1.0) requestAnimationFrame(animate)
    }

    requestAnimationFrame(animate)
  }

  // ---- Статистика ----

  private _updateStats(delta: number): void {
    this._stats.fps       = Math.round(1 / delta)
    this._stats.frameTime = Math.round(delta * 1000)
    this._stats.totalTiles = this._tiles.size
    this._stats.drawCalls  = this._renderer.info.render.calls
    this._stats.triangles  = this._renderer.info.render.triangles
  }
}

// ----------------------------------------------------------------
// ТИПИ
// ----------------------------------------------------------------

export interface RenderStats {
  fps:          number
  frameTime:    number    // мс
  visibleTiles: number
  totalTiles:   number
  triangles:    number
  drawCalls:    number
  quadtree:     QuadtreeStats
}

// ----------------------------------------------------------------
// УТИЛІТИ
// ----------------------------------------------------------------

async function _isWebGPUSupported(): Promise<boolean> {
  try {
    if (!('gpu' in navigator)) return false
    const adapter = await (navigator as any).gpu.requestAdapter()
    return adapter !== null
  } catch {
    return false
  }
}

function _easeInOutCubic(t: number): number {
  return t < 0.5
    ? 4 * t * t * t
    : 1 - (-2 * t + 2) ** 3 / 2
  }
