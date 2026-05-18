  /**
 * GeoEngine — GeoRenderer (оновлений)
 * Підключає TerrainRenderer та Pipeline до основного render loop.
 *
 * Зміни відносно попередньої версії:
 * - Використовує WebGPUPipeline / ThreeJSPipeline
 * - TerrainRenderer керує матеріалами тайлів
 * - Render loop викликає pipeline.beginFrame/endFrame
 * - Stats збираються з TerrainRenderer
 */

import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { BBox, TileXYZ, LODLevel } from '../../shared-types/src/geo'
import { TerrainTile }      from '../terrain/TerrainTile'
import { QuadtreeLOD }      from '../terrain/Quadtree'
import { llhToWorld, haversineDistance } from '../geo/Coords'
import { TerrainRenderer, createTerrainMaterial } from './TerrainRenderer'
import { WebGPUPipeline, ThreeJSPipeline }        from './Pipeline'
import { LRUCache } from '../utils/lruCache'
import type { RenderStats, RendererState, GeoRendererOptions } from './GeoRenderer.types'

export class GeoRenderer {

  // Three.js core
  private _threeRenderer!: THREE.WebGLRenderer
  private _scene!:         THREE.Scene
  private _camera!:        THREE.PerspectiveCamera
  private _controls!:      OrbitControls

  // Pipeline (WebGPU або Three.js)
  private _pipeline:       WebGPUPipeline | ThreeJSPipeline | null = null
  private _isWebGPU:       boolean = false

  // Terrain renderer
  private _terrainRenderer: TerrainRenderer | null = null

  // LOD система
  private _quadtree!:      QuadtreeLOD

  // Тайли
  private _tiles:          LRUCache<string, TerrainTile>
  private _pendingKeys:    Set<string> = new Set()
  private _visibleTiles:   TerrainTile[] = []

  // WebSocket
  private _ws:             WebSocket | null = null

  // Стан
  private _state:          RendererState = 'idle'
  private _frameId:        number | null = null
  private _clock:          THREE.Clock   = new THREE.Clock()

  // Config
  private _opts:           Required<GeoRendererOptions>
  private _originLat:      number
  private _originLon:      number

  // Frustum
  private _frustum:        THREE.Frustum = new THREE.Frustum()
  private _projMatrix:     THREE.Matrix4 = new THREE.Matrix4()

  // Stats
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

    // LRU кеш тайлів (64 тайли)
    this._tiles = new LRUCache<string, TerrainTile>({
      maxSize: 64,
      ttlMs:   30_000,
      onEvict: (_, tile) => {
        if (tile.mesh?.parent) this._scene?.remove(tile.mesh)
        tile.dispose()
      },
    })
  }

  // ---- Ініціалізація ----

  async init(): Promise<void> {
    if (this._state !== 'idle') throw new Error('Already initialized')
    this._state = 'initializing'

    try {
      // Three.js renderer (завжди потрібен)
      await this._initThreeRenderer()
      this._initScene()
      this._initCamera()
      this._initLights()
      this._initControls()
      this._initLOD()

      const canvas = this._threeRenderer.domElement

      // Спробувати WebGPU pipeline
      if (this._opts.renderer !== 'webgl2') {
        const gpuPipeline = new WebGPUPipeline({
          canvas,
          msaaSamples: 4,
        })
        const ok = await gpuPipeline.init()

        if (ok) {
          this._pipeline = gpuPipeline
          this._isWebGPU  = true
        }
      }

      // Fallback до Three.js pipeline
      if (!this._isWebGPU) {
        this._pipeline = new ThreeJSPipeline(
          this._threeRenderer, this._scene, this._camera
        )
      }

      // TerrainRenderer
      this._terrainRenderer = new TerrainRenderer({
        device:        this._isWebGPU
          ? (this._pipeline as WebGPUPipeline).device
          : null,
        threeRenderer: this._threeRenderer,
        scene:         this._scene,
        camera:        this._camera,
        terrain: {
          seaLevel:    0,
          snowStart:   2000,
          rockStart:   1200,
        },
      })
      await this._terrainRenderer.init()

      // WebSocket
      this._initWebSocket()
      this._initResizeObserver()

      this._state = 'running'
      console.log('[GeoRenderer] ready', {
        mode:   this._isWebGPU ? 'WebGPU' : 'WebGL2',
        origin: `${this._originLat}, ${this._originLon}`,
      })

    } catch (err) {
      this._state = 'error'
      throw err
    }
  }

  // ---- Render loop ----

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

  private _loop(): void {
    this._frameId = requestAnimationFrame(() => this._loop())

    const delta = this._clock.getDelta()
    this._update(delta)

    if (this._isWebGPU && this._pipeline instanceof WebGPUPipeline) {
      // WebGPU render path
      const pass = this._pipeline.beginFrame()
      if (pass && this._terrainRenderer) {
        this._terrainRenderer.renderTiles(this._visibleTiles, pass)
      }
      this._pipeline.endFrame()
    } else {
      // Three.js render path (TerrainRenderer оновлює матеріали)
      if (this._terrainRenderer) {
        this._terrainRenderer.renderTiles(this._visibleTiles)
      }
      this._pipeline?.endFrame()
    }

    this._updateStats(delta)
  }

  private _update(delta: number): void {
    this._controls.update()

    // Terrain renderer update (sun direction, etc.)
    this._terrainRenderer?.update(delta)

    // Frustum update
    this._projMatrix.multiplyMatrices(
      this._camera.projectionMatrix,
      this._camera.matrixWorldInverse,
    )
    this._frustum.setFromProjectionMatrix(this._projMatrix)

    // Camera lat/lon від world position
    const camWorld = this._camera.position
    const R        = 6_378_137.0
    const cosLat   = Math.cos(this._originLat * Math.PI / 180)
    const camLat   = this._originLat + (-camWorld.z) / 111_320
    const camLon   = this._originLon + camWorld.x / (cosLat * R * Math.PI / 180)
    const camAlt   = camWorld.y

    // LOD update
    const visibleNodes = this._quadtree.getVisibleTiles(
      camLat, camLon, camAlt, this._frustum,
    )

    this._stats.quadtree    = this._quadtree.getStats()
    this._stats.visibleTiles = visibleNodes.length

    // Список тайлів для рендерингу
    this._visibleTiles = []
    for (const { tile, lodLevel } of visibleNodes) {
      const key = `${tile.z}/${tile.x}/${tile.y}`
      const cached = this._tiles.get(key)

      if (cached?.isReady) {
        cached.touch()
        this._visibleTiles.push(cached)

        // Three.js: додати в сцену якщо ще немає
        if (!this._isWebGPU && cached.mesh && !cached.mesh.parent) {
          this._scene.add(cached.mesh)
        }
      } else if (!this._pendingKeys.has(key)) {
        this._requestTile(tile, lodLevel)
      }
    }

    // Three.js: прибрати невидимі тайли зі сцени
    if (!this._isWebGPU) {
      const visibleKeys = new Set(
        visibleNodes.map(n => `${n.tile.z}/${n.tile.x}/${n.tile.y}`)
      )
      for (const [key, tile] of this._tiles.entries()) {
        if (tile.mesh?.parent && !visibleKeys.has(key)) {
          this._scene.remove(tile.mesh)
        }
      }
    }
  }

  // ---- Camera ----

  flyTo(options: {
    lat: number; lon: number; altitude: number; duration?: number
  }): void {
    const { lat, lon, altitude, duration = 2000 } = options
    const target = llhToWorld(lat, lon, 0,        this._originLat, this._originLon)
    const camPos = llhToWorld(lat, lon, altitude, this._originLat, this._originLon)

    this._animateFlyTo(
      this._camera.position.clone(),
      new THREE.Vector3(camPos.x, camPos.y, camPos.z),
      new THREE.Vector3(target.x, target.y, target.z),
      duration,
    )
  }

  setCameraPosition(lat: number, lon: number, altitude: number): void {
    const pos    = llhToWorld(lat, lon, altitude, this._originLat, this._originLon)
    const target = llhToWorld(lat, lon, 0,        this._originLat, this._originLon)
    this._camera.position.set(pos.x, pos.y, pos.z)
    this._controls.target.set(target.x, target.y, target.z)
    this._controls.update()
  }

  setTimeOfDay(hours: number): void {
    this._terrainRenderer?.setTimeOfDay(hours)
  }

  setWireframe(enabled: boolean): void {
    this._terrainRenderer?.setWireframe(enabled)
  }

  // ---- Tile streaming ----

  private _requestTile(tile: TileXYZ, lodLevel: LODLevel): void {
    const key = `${tile.z}/${tile.x}/${tile.y}`
    this._pendingKeys.add(key)

    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) {
      this._pendingKeys.delete(key)
      return
    }

    this._ws.send(JSON.stringify({
      type:      'request_tile',
      id:        `${key}-${Date.now()}`,
      timestamp: Date.now(),
      payload: {
        tile,
        source:       'terrarium',
        max_vertices: this._getMaxVertsByLOD(lodLevel),
        skirt_height_m: 200,
      },
    }))
  }

  private _getMaxVertsByLOD(lod: LODLevel): number {
    const map: Record<number, number> = {
      0: 65_536, 1: 16_384, 2: 4_096,
      3: 1_024,  4: 256,    5: 64,
    }
    return map[lod] ?? 4_096
  }

  private _onTileReceived(data: any): void {
    const tileKey = this._pendingKeys.values().next().value
    if (!tileKey) return
    this._pendingKeys.delete(tileKey)

    const terrainTile = new TerrainTile(
      { x: 0, y: 0, z: data.lod_level ?? 0 },
      data.lod_level ?? 0,
    )

    const material = this._terrainRenderer?.material
      ?? createTerrainMaterial()

    terrainTile.buildFromServerData(data, material)

    if (terrainTile.isReady) {
      this._tiles.set(tileKey, terrainTile)
    }
  }

  // ---- Private init helpers ----

  private async _initThreeRenderer(): Promise<void> {
    const canvas = typeof this._opts.canvas === 'string'
      ? document.querySelector<HTMLCanvasElement>(this._opts.canvas)!
      : this._opts.canvas

    this._threeRenderer = new THREE.WebGLRenderer({
      canvas,
      antialias:              true,
      logarithmicDepthBuffer: true,
      alpha:                  false,
    })

    this._threeRenderer.setPixelRatio(window.devicePixelRatio)
    this._threeRenderer.setSize(canvas.clientWidth, canvas.clientHeight)
    this._threeRenderer.shadowMap.enabled = true
    this._threeRenderer.toneMapping       = THREE.ACESFilmicToneMapping
    this._threeRenderer.toneMappingExposure = 1.0
    this._threeRenderer.outputColorSpace  = THREE.SRGBColorSpace
  }

  private _initScene(): void {
    this._scene = new THREE.Scene()
    this._scene.background = new THREE.Color(0x87ceeb)
    this._scene.fog         = new THREE.FogExp2(0xcce8ff, 0.000012)
  }

  private _initCamera(): void {
    const el = this._threeRenderer.domElement
    this._camera = new THREE.PerspectiveCamera(
      60,
      el.clientWidth / el.clientHeight,
      1,
      10_000_000,
    )
    const { lat, lon, altitude } = this._opts.initialCamera
    const pos = llhToWorld(lat, lon, altitude, this._originLat, this._originLon)
    this._camera.position.set(pos.x, pos.y, pos.z)
    this._camera.lookAt(0, 0, 0)
  }

  private _initLights(): void {
    this._scene.add(new THREE.AmbientLight(0xffffff, 0.4))

    const sun = new THREE.DirectionalLight(0xfffaed, 1.2)
    sun.position.set(5000, 8000, 3000)
    sun.castShadow = true
    sun.shadow.mapSize.set(2048, 2048)
    sun.shadow.camera.near   = 1
    sun.shadow.camera.far    = 50_000
    const sc = sun.shadow.camera as THREE.OrthographicCamera
    sc.left = sc.bottom = -20_000
    sc.right = sc.top   =  20_000
    this._scene.add(sun)

    this._scene.add(new THREE.HemisphereLight(0x87ceeb, 0x4a7c3f, 0.3))
  }

  private _initControls(): void {
    this._controls = new OrbitControls(this._camera, this._threeRenderer.domElement)
    this._controls.enableDamping    = true
    this._controls.dampingFactor    = 0.05
    this._controls.screenSpacePanning = false
    this._controls.minDistance      = 100
    this._controls.maxDistance      = 5_000_000
    this._controls.maxPolarAngle    = Math.PI / 2 - 0.01
  }

  private _initLOD(): void {
    this._quadtree = new QuadtreeLOD({
      rootZoom: 4,
      maxZoom:  this._opts.maxLODZoom,
    })
  }

  private _initWebSocket(): void {
    const connect = (): void => {
      this._ws = new WebSocket(this._opts.serverUrl)
      this._ws.onopen    = () => console.log('[GeoRenderer] WS connected')
      this._ws.onmessage = (e: MessageEvent) => {
        try {
          const msg = JSON.parse(e.data as string)
          if (msg.type === 'response_tile') this._onTileReceived(msg.payload)
        } catch { /* ignore parse errors */ }
      }
      this._ws.onclose = () => setTimeout(connect, 3000)
    }
    connect()
  }

  private _initResizeObserver(): void {
    const el = this._threeRenderer.domElement
    new ResizeObserver(() => {
      const w = el.clientWidth, h = el.clientHeight
      this._camera.aspect = w / h
      this._camera.updateProjectionMatrix()
      this._threeRenderer.setSize(w, h)
      this._pipeline?.resize(w, h)
    }).observe(el)
  }

  private _animateFlyTo(
    from: THREE.Vector3, to: THREE.Vector3,
    target: THREE.Vector3, duration: number,
  ): void {
    const start = performance.now()
    const fromTarget = this._controls.target.clone()

    const animate = (): void => {
      const t      = Math.min((performance.now() - start) / duration, 1)
      const eased  = t < 0.5 ? 4*t*t*t : 1 - (-2*t+2)**3/2
      this._camera.position.lerpVectors(from, to, eased)
      this._controls.target.lerpVectors(fromTarget, target, eased)
      this._controls.update()
      if (t < 1) requestAnimationFrame(animate)
    }
    requestAnimationFrame(animate)
  }

  private _updateStats(delta: number): void {
    this._stats.fps       = Math.round(1 / delta)
    this._stats.frameTime = Math.round(delta * 1000)
    this._stats.totalTiles = this._tiles.size

    if (this._terrainRenderer) {
      const ts = this._terrainRenderer.stats
      this._stats.drawCalls = ts.drawCalls
      this._stats.triangles = ts.triangles
    } else {
      this._stats.drawCalls = this._threeRenderer.info.render.calls
      this._stats.triangles = this._threeRenderer.info.render.triangles
    }
  }

  // ---- Public getters ----

  get state():  RendererState { return this._state }
  get stats():  RenderStats   { return { ...this._stats } }
  get scene():  THREE.Scene   { return this._scene }
  get camera(): THREE.PerspectiveCamera { return this._camera }
  get isWebGPU(): boolean     { return this._isWebGPU }

  dispose(): void {
    this.stop()
    this._ws?.close()
    this._tiles.clear()
    this._terrainRenderer?.dispose()
    this._pipeline?.dispose()
    this._state = 'idle'
  }
                              }
