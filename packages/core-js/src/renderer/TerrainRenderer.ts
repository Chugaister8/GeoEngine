/**
 * GeoEngine — TerrainRenderer
 * Окремий WebGPU render pass для терейну.
 *
 * Відповідає за:
 * - Створення WebGPU pipeline (vertex + fragment)
 * - Управління uniform буферами (camera, terrain, lighting, fog)
 * - Bind groups для текстур
 * - Render pass для всіх TerrainTile
 * - LOD morph factor оновлення
 * - Frustum culling на CPU перед draw call
 *
 * Використовує terrain.wgsl шейдер.
 * Fallback: Three.js MeshStandardMaterial для WebGL2.
 */

import * as THREE from 'three'
import type { TerrainTile } from '../terrain/TerrainTile'
import {
  terrainWGSL,
  createUniformBuffer,
  updateUniformBuffer,
  packCameraUniforms,
  packTerrainUniforms,
  packFogUniforms,
} from '../shaders/index'
import { LRUCache } from '../utils/lruCache'

// ----------------------------------------------------------------
// ТИПИ
// ----------------------------------------------------------------

export interface TerrainRendererOptions {
  /** WebGPU device (якщо є) */
  device?:          GPUDevice | null

  /** Three.js renderer (завжди є — fallback) */
  threeRenderer:    THREE.WebGLRenderer

  /** Three.js scene */
  scene:            THREE.Scene

  /** Three.js camera */
  camera:           THREE.PerspectiveCamera

  /** Параметри терейну */
  terrain?: {
    seaLevel?:      number   // метри, default 0
    grassEnd?:      number   // метри, default 800
    rockStart?:     number   // метри, default 1200
    snowStart?:     number   // метри, default 2000
    slopeRockDeg?:  number   // градуси, default 35
    slopeSNowDeg?:  number   // градуси, default 55
    uvScale?:       number   // default 8
  }

  /** Fog параметри */
  fog?: {
    color?:   [number, number, number]
    density?: number
    start?:   number
    end?:     number
    height?:  number
  }
}

export interface RenderStats {
  drawCalls:      number
  triangles:      number
  tilesRendered:  number
  tilesTotal:     number
  frameTimeMs:    number
  lodDistribution: Record<number, number>  // lod → count
}

// ----------------------------------------------------------------
// TERRAIN MATERIAL FACTORY
// ----------------------------------------------------------------

/**
 * Створює Three.js матеріал для терейну.
 * Використовується як fallback (WebGL2) та для WebGPU до
 * повного підключення кастомного pipeline.
 */
export function createTerrainMaterial(options: {
  wireframe?: boolean
  vertexColors?: boolean
} = {}): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color:      0x7a9e6e,
    roughness:  0.9,
    metalness:  0.0,
    wireframe:  options.wireframe ?? false,
    vertexColors: options.vertexColors ?? false,
    side:       THREE.FrontSide,
  })
}

/**
 * Elevation-based vertex shader (Three.js ShaderMaterial).
 * Фарбує терейн за висотою — grass/rock/snow.
 * Використовується до підключення повного WGSL pipeline.
 */
export function createElevationMaterial(params: {
  minElevation?: number
  maxElevation?: number
  seaLevel?:     number
  snowStart?:    number
} = {}): THREE.ShaderMaterial {
  const {
    minElevation = 0,
    maxElevation = 3000,
    seaLevel     = 0,
    snowStart    = 2000,
  } = params

  return new THREE.ShaderMaterial({
    uniforms: {
      u_minElev:  { value: minElevation },
      u_maxElev:  { value: maxElevation },
      u_seaLevel: { value: seaLevel },
      u_snowStart:{ value: snowStart },
      u_sunDir:   { value: new THREE.Vector3(0.5, 0.8, 0.3).normalize() },
      u_ambient:  { value: 0.35 },
    },
    vertexShader: `
      varying float v_elevation;
      varying vec3  v_normal;
      varying vec3  v_worldPos;

      void main() {
        v_elevation = position.y;
        v_normal    = normalize(normalMatrix * normal);
        v_worldPos  = (modelMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform float u_minElev;
      uniform float u_maxElev;
      uniform float u_seaLevel;
      uniform float u_snowStart;
      uniform vec3  u_sunDir;
      uniform float u_ambient;

      varying float v_elevation;
      varying vec3  v_normal;
      varying vec3  v_worldPos;

      vec3 grass = vec3(0.38, 0.55, 0.28);
      vec3 rock  = vec3(0.52, 0.48, 0.42);
      vec3 snow  = vec3(0.95, 0.96, 0.98);
      vec3 sand  = vec3(0.82, 0.76, 0.58);
      vec3 deep  = vec3(0.20, 0.35, 0.55);

      void main() {
        float norm  = clamp((v_elevation - u_minElev) / max(u_maxElev - u_minElev, 1.0), 0.0, 1.0);
        float slope = 1.0 - abs(dot(v_normal, vec3(0.0, 1.0, 0.0)));

        // Base color by elevation
        vec3 color = grass;
        if (v_elevation < u_seaLevel) {
          color = mix(sand, deep, clamp((u_seaLevel - v_elevation) / 20.0, 0.0, 1.0));
        }
        float rockFactor = smoothstep(0.4, 0.8, norm) + smoothstep(0.5, 0.8, slope);
        color = mix(color, rock, clamp(rockFactor, 0.0, 1.0));
        float snowFactor = smoothstep(0.75, 0.95, norm) * (1.0 - slope * 1.5);
        color = mix(color, snow, clamp(snowFactor, 0.0, 1.0));

        // Lighting (Lambert)
        float diff = max(0.0, dot(v_normal, normalize(u_sunDir)));
        float lit  = u_ambient + (1.0 - u_ambient) * diff;

        // Slight AO from elevation
        float ao = mix(0.85, 1.0, norm);

        gl_FragColor = vec4(color * lit * ao, 1.0);
      }
    `,
    side: THREE.FrontSide,
  })
}

// ----------------------------------------------------------------
// WEBGPU PIPELINE MANAGER
// ----------------------------------------------------------------

/**
 * Управляє WebGPU pipeline для терейну.
 * Інкапсулює всю WebGPU-специфічну логіку.
 */
class WebGPUTerrainPipeline {
  private _device:     GPUDevice
  private _pipeline:   GPURenderPipeline | null = null
  private _ready:      boolean = false

  // Uniform buffers
  private _camBuffer:     GPUBuffer | null = null
  private _terrainBuffer: GPUBuffer | null = null
  private _lightBuffer:   GPUBuffer | null = null
  private _fogBuffer:     GPUBuffer | null = null

  // Bind group layouts
  private _globalBGL:  GPUBindGroupLayout | null = null
  private _textureBGL: GPUBindGroupLayout | null = null

  // Глобальний bind group (camera + terrain + lighting + fog)
  private _globalBG:   GPUBindGroup | null = null

  // Дефолтні текстури (1×1 пікс заглушки)
  private _defaultSampler:  GPUSampler | null = null
  private _defaultTexture:  GPUTexture | null = null
  private _defaultTextureView: GPUTextureView | null = null

  constructor(device: GPUDevice) {
    this._device = device
  }

  async init(format: GPUTextureFormat): Promise<void> {
    this._createUniformBuffers()
    this._createDefaultTextures()
    await this._createPipeline(format)
    this._createGlobalBindGroup()
    this._ready = true
    console.log('[WebGPUTerrainPipeline] ініціалізований')
  }

  get isReady(): boolean { return this._ready }

  // ---- Оновлення uniforms ----

  updateCamera(
    camera:     THREE.PerspectiveCamera,
    canvas:     HTMLCanvasElement,
  ): void {
    if (!this._camBuffer) return

    const view     = new Float32Array(camera.matrixWorldInverse.elements)
    const proj     = new Float32Array(camera.projectionMatrix.elements)
    const viewProj = new Float32Array(
      new THREE.Matrix4()
        .multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
        .elements
    )
    const pos = camera.position

    const data = packCameraUniforms(
      view, proj, viewProj,
      [pos.x, pos.y, pos.z],
      camera.near,
      camera.far,
    )
    this._device.queue.writeBuffer(this._camBuffer, 0, data)
  }

  updateTerrain(params: {
    minElevation: number
    maxElevation: number
    lodLevel:     number
    morphFactor:  number
  }): void {
    if (!this._terrainBuffer) return

    const data = packTerrainUniforms({
      minElevation:  params.minElevation,
      maxElevation:  params.maxElevation,
      lodLevel:      params.lodLevel,
      morphFactor:   params.morphFactor,
      scale:         1.0,
      uvScale:       8.0,
      seaLevel:      0.0,
      grassEnd:      800.0,
      rockStart:     1200.0,
      snowStart:     2000.0,
      slopeRockDeg:  35.0,
      slopeSNowDeg:  55.0,
    })
    this._device.queue.writeBuffer(this._terrainBuffer, 0, data)
  }

  updateLighting(sunDirection: THREE.Vector3, intensity = 1.2): void {
    if (!this._lightBuffer) return

    const data = new Float32Array([
      sunDirection.x, sunDirection.y, sunDirection.z,
      intensity,
      1.0, 0.98, 0.92,   // sun color (warm white)
      0.35,              // ambient
      0.5,               // shadow softness
      0, 0, 0,           // padding
    ])
    this._device.queue.writeBuffer(this._lightBuffer, 0, data)
  }

  updateFog(params: {
    color:   [number, number, number]
    density: number
    start:   number
    end:     number
    height:  number
  }): void {
    if (!this._fogBuffer) return

    const data = packFogUniforms(params)
    this._device.queue.writeBuffer(this._fogBuffer, 0, data)
  }

  // ---- Приватні методи ----

  private _createUniformBuffers(): void {
    const d = this._device

    // Camera: view(16) + proj(16) + viewProj(16) + pos(3) + near(1) + far(1) + pad(3) = 56 floats
    this._camBuffer     = createUniformBuffer(d, 56 * 4, 'terrain-camera')

    // Terrain: 14 floats
    this._terrainBuffer = createUniformBuffer(d, 14 * 4, 'terrain-params')

    // Lighting: 12 floats
    this._lightBuffer   = createUniformBuffer(d, 12 * 4, 'terrain-lighting')

    // Fog: 8 floats
    this._fogBuffer     = createUniformBuffer(d, 8 * 4, 'terrain-fog')
  }

  private _createDefaultTextures(): void {
    const d = this._device

    this._defaultSampler = d.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    })

    // 1×1 rgba8unorm texture
    this._defaultTexture = d.createTexture({
      size:   [1, 1],
      format: 'rgba8unorm',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })

    // Зелений колір (трава)
    d.queue.writeTexture(
      { texture: this._defaultTexture },
      new Uint8Array([100, 140, 70, 255]),
      { bytesPerRow: 4 },
      [1, 1],
    )

    this._defaultTextureView = this._defaultTexture.createView()
  }

  private async _createPipeline(format: GPUTextureFormat): Promise<void> {
    const d = this._device

    // Компіляція шейдера
    const shaderModule = d.createShaderModule({
      label: 'GeoEngine/terrain',
      code:  terrainWGSL as string,
    })

    // Перевірка помилок компіляції
    if ('getCompilationInfo' in shaderModule) {
      const info = await (shaderModule as any).getCompilationInfo()
      for (const msg of info.messages) {
        if (msg.type === 'error') {
          throw new Error(`WGSL compile error: ${msg.message} (line ${msg.lineNum})`)
        }
        if (msg.type === 'warning') {
          console.warn('[TerrainShader]', msg.message)
        }
      }
    }

    // Bind group layouts
    this._globalBGL = d.createBindGroupLayout({
      label: 'terrain-global-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' } },  // camera
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' } },  // terrain
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' } },  // lighting
        { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' } },  // fog
      ],
    })

    this._textureBGL = d.createBindGroupLayout({
      label: 'terrain-texture-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: 'filtering' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' } },  // grass
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' } },  // rock
        { binding: 3, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' } },  // snow
        { binding: 4, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' } },  // sand
        { binding: 5, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' } },  // normal map
        { binding: 6, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' } },  // satellite
      ],
    })

    const pipelineLayout = d.createPipelineLayout({
      label: 'terrain-pipeline-layout',
      bindGroupLayouts: [this._globalBGL, this._textureBGL],
    })

    // Render pipeline
    this._pipeline = d.createRenderPipeline({
      label:  'terrain-render-pipeline',
      layout: pipelineLayout,

      vertex: {
        module:     shaderModule,
        entryPoint: 'vs_main',
        buffers: [
          {
            // Position (vec3)
            arrayStride: 12,
            stepMode:    'vertex',
            attributes:  [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
          },
          {
            // UV (vec2)
            arrayStride: 8,
            stepMode:    'vertex',
            attributes:  [{ shaderLocation: 1, offset: 0, format: 'float32x2' }],
          },
          {
            // Normal (vec3)
            arrayStride: 12,
            stepMode:    'vertex',
            attributes:  [{ shaderLocation: 2, offset: 0, format: 'float32x3' }],
          },
        ],
      },

      fragment: {
        module:     shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format }],
      },

      primitive: {
        topology:  'triangle-list',
        cullMode:  'back',
        frontFace: 'ccw',
      },

      depthStencil: {
        format:            'depth24plus',
        depthWriteEnabled: true,
        depthCompare:      'less',
      },

      multisample: { count: 4 },  // MSAA 4×
    })
  }

  private _createGlobalBindGroup(): void {
    if (!this._globalBGL || !this._camBuffer ||
        !this._terrainBuffer || !this._lightBuffer || !this._fogBuffer) return

    this._globalBG = this._device.createBindGroup({
      label:  'terrain-global-bg',
      layout: this._globalBGL,
      entries: [
        { binding: 0, resource: { buffer: this._camBuffer } },
        { binding: 1, resource: { buffer: this._terrainBuffer } },
        { binding: 2, resource: { buffer: this._lightBuffer } },
        { binding: 3, resource: { buffer: this._fogBuffer } },
      ],
    })
  }

  createTextureBindGroup(textures: {
    grass?:     GPUTextureView
    rock?:      GPUTextureView
    snow?:      GPUTextureView
    sand?:      GPUTextureView
    normalMap?: GPUTextureView
    satellite?: GPUTextureView
  }): GPUBindGroup {
    const dv = this._defaultTextureView!

    return this._device.createBindGroup({
      label:  'terrain-texture-bg',
      layout: this._textureBGL!,
      entries: [
        { binding: 0, resource: this._defaultSampler! },
        { binding: 1, resource: textures.grass     ?? dv },
        { binding: 2, resource: textures.rock      ?? dv },
        { binding: 3, resource: textures.snow      ?? dv },
        { binding: 4, resource: textures.sand      ?? dv },
        { binding: 5, resource: textures.normalMap ?? dv },
        { binding: 6, resource: textures.satellite ?? dv },
      ],
    })
  }

  // ---- GPU Buffers для тайлу ----

  createTileBuffers(tile: TerrainTile): TileGPUBuffers | null {
    if (!tile.isReady || !tile.mesh) return null

    const geo  = tile.mesh.geometry
    const d    = this._device

    const posAttr  = geo.attributes['position'] as THREE.BufferAttribute
    const uvAttr   = geo.attributes['uv']       as THREE.BufferAttribute
    const normAttr = geo.attributes['normal']   as THREE.BufferAttribute
    const idxAttr  = geo.index

    if (!posAttr || !uvAttr || !normAttr || !idxAttr) return null

    const posBuf = _uploadBuffer(d, posAttr.array as Float32Array,
      GPUBufferUsage.VERTEX, 'pos')
    const uvBuf  = _uploadBuffer(d, uvAttr.array  as Float32Array,
      GPUBufferUsage.VERTEX, 'uv')
    const nrmBuf = _uploadBuffer(d, normAttr.array as Float32Array,
      GPUBufferUsage.VERTEX, 'normal')
    const idxBuf = _uploadBuffer(d, idxAttr.array  as Uint32Array,
      GPUBufferUsage.INDEX, 'index')

    return {
      position:    posBuf,
      uv:          uvBuf,
      normal:      nrmBuf,
      index:       idxBuf,
      indexCount:  idxAttr.count,
    }
  }

  // ---- Draw call ----

  recordDraw(
    pass:      GPURenderPassEncoder,
    buffers:   TileGPUBuffers,
    textureBG: GPUBindGroup,
  ): void {
    if (!this._pipeline || !this._globalBG) return

    pass.setPipeline(this._pipeline)
    pass.setBindGroup(0, this._globalBG)
    pass.setBindGroup(1, textureBG)
    pass.setVertexBuffer(0, buffers.position)
    pass.setVertexBuffer(1, buffers.uv)
    pass.setVertexBuffer(2, buffers.normal)
    pass.setIndexBuffer(buffers.index, 'uint32')
    pass.drawIndexed(buffers.indexCount)
  }

  dispose(): void {
    this._camBuffer?.destroy()
    this._terrainBuffer?.destroy()
    this._lightBuffer?.destroy()
    this._fogBuffer?.destroy()
    this._defaultTexture?.destroy()
    this._ready = false
  }
}

interface TileGPUBuffers {
  position:   GPUBuffer
  uv:         GPUBuffer
  normal:     GPUBuffer
  index:      GPUBuffer
  indexCount: number
}

// ----------------------------------------------------------------
// TERRAIN RENDERER (головний клас)
// ----------------------------------------------------------------

export class TerrainRenderer {
  /**
   * Рендерер терейну що підтримує обидва шляхи:
   *
   * WebGPU path (якщо device доступний):
   *   custom WGSL pipeline → elevation colormap + PBR lighting
   *
   * WebGL2 fallback (Three.js):
   *   ShaderMaterial з elevation colormap
   *   Все через Three.js render loop
   *
   * Публічний API однаковий для обох — рендерер вибирає
   * правильний шлях автоматично.
   */

  private _opts:       TerrainRendererOptions
  private _gpuPipeline: WebGPUTerrainPipeline | null = null
  private _isWebGPU:   boolean = false

  // Three.js матеріал (shared між всіма тайлами)
  private _material:   THREE.Material

  // GPU буфери тайлів (WebGPU path)
  private _tileBuffers: LRUCache<string, TileGPUBuffers>

  // Дефолтний texture bind group
  private _defaultTextureBG: GPUBindGroup | null = null

  // Статистика
  private _stats: RenderStats = {
    drawCalls: 0, triangles: 0,
    tilesRendered: 0, tilesTotal: 0,
    frameTimeMs: 0, lodDistribution: {},
  }

  // Sun direction (оновлюється кожен кадр)
  private _sunDir = new THREE.Vector3(0.5, 0.8, 0.3).normalize()

  // Параметри анімації sun
  private _timeOfDay = 0.5   // [0..1]

  constructor(options: TerrainRendererOptions) {
    this._opts = options

    // Матеріал для Three.js fallback
    this._material = createElevationMaterial({
      seaLevel:  options.terrain?.seaLevel  ?? 0,
      snowStart: options.terrain?.snowStart ?? 2000,
    })

    // LRU кеш GPU буферів (64 тайли)
    this._tileBuffers = new LRUCache<string, TileGPUBuffers>({
      maxSize: 64,
      onEvict: (_, bufs) => {
        bufs.position.destroy()
        bufs.uv.destroy()
        bufs.normal.destroy()
        bufs.index.destroy()
      },
    })
  }

  // ---- Ініціалізація ----

  async init(): Promise<void> {
    const { device, threeRenderer } = this._opts

    if (device) {
      try {
        this._gpuPipeline = new WebGPUTerrainPipeline(device)
        const format = navigator.gpu?.getPreferredCanvasFormat?.() ?? 'bgra8unorm'
        await this._gpuPipeline.init(format)
        this._isWebGPU = true
        console.log('[TerrainRenderer] WebGPU pipeline активний')

        // Дефолтний texture bind group
        this._defaultTextureBG = this._gpuPipeline.createTextureBindGroup({})

        // Початкові uniforms
        this._gpuPipeline.updateLighting(this._sunDir)
        this._gpuPipeline.updateFog(this._buildFogParams())
        this._gpuPipeline.updateTerrain({
          minElevation: 0, maxElevation: 3000,
          lodLevel: 0, morphFactor: 0,
        })

      } catch (err) {
        console.warn('[TerrainRenderer] WebGPU pipeline failed, fallback to WebGL2:', err)
        this._isWebGPU    = false
        this._gpuPipeline = null
      }
    } else {
      console.log('[TerrainRenderer] WebGL2 fallback (Three.js ShaderMaterial)')
    }
  }

  // ---- Оновлення кожен кадр ----

  /**
   * Оновити стан рендерера.
   * Викликати перед render() кожен кадр.
   */
  update(deltaTime: number): void {
    // Оновити sun direction за часом доби
    const angle = (this._timeOfDay - 0.25) * Math.PI * 2
    this._sunDir.set(
      Math.cos(angle),
      Math.sin(angle) * 0.8 + 0.2,
      Math.sin(angle) * 0.3,
    ).normalize()

    // Оновити шейдер uniform (Three.js material)
    if (this._material instanceof THREE.ShaderMaterial) {
      this._material.uniforms['u_sunDir']?.value?.copy(this._sunDir)
    }

    // Оновити WebGPU uniforms
    if (this._isWebGPU && this._gpuPipeline?.isReady) {
      this._gpuPipeline.updateCamera(
        this._opts.camera,
        this._opts.threeRenderer.domElement,
      )
      this._gpuPipeline.updateLighting(this._sunDir)
    }
  }

  /**
   * Рендерити список тайлів.
   *
   * WebGPU: використовує custom pipeline + WGSL шейдер
   * WebGL2: тайли вже у Three.js scene → Three.js render їх автоматично
   */
  renderTiles(
    tiles:        TerrainTile[],
    passEncoder?: GPURenderPassEncoder,   // тільки для WebGPU
  ): void {
    const t0 = performance.now()

    this._stats.drawCalls     = 0
    this._stats.triangles     = 0
    this._stats.tilesRendered = 0
    this._stats.tilesTotal    = tiles.length
    this._stats.lodDistribution = {}

    for (const tile of tiles) {
      if (!tile.isReady) continue

      // LOD статистика
      const lod = tile.lodLevel
      this._stats.lodDistribution[lod] = (this._stats.lodDistribution[lod] ?? 0) + 1

      if (this._isWebGPU && passEncoder && this._gpuPipeline?.isReady) {
        // WebGPU path
        this._renderTileWebGPU(tile, passEncoder)
      } else {
        // WebGL2 path — матеріал вже встановлений при buildFromHeightmap/ServerData
        this._ensureThreeMaterial(tile)
        this._stats.drawCalls++
        this._stats.triangles += tile.triangleCount
        this._stats.tilesRendered++
      }
    }

    this._stats.frameTimeMs = performance.now() - t0
  }

  // ---- WebGPU render ----

  private _renderTileWebGPU(
    tile:    TerrainTile,
    pass:    GPURenderPassEncoder,
  ): void {
    const key = tile.tileKey

    // Отримати або створити GPU буфери
    let buffers = this._tileBuffers.get(key)
    if (!buffers) {
      buffers = this._gpuPipeline!.createTileBuffers(tile)
      if (!buffers) return
      this._tileBuffers.set(key, buffers)
    }

    // Оновити terrain uniforms для цього тайлу
    this._gpuPipeline!.updateTerrain({
      minElevation: tile.minElevation,
      maxElevation: tile.maxElevation,
      lodLevel:     tile.lodLevel,
      morphFactor:  0.0,
    })

    // Draw
    this._gpuPipeline!.recordDraw(
      pass,
      buffers,
      this._defaultTextureBG!,
    )

    this._stats.drawCalls++
    this._stats.triangles     += tile.triangleCount
    this._stats.tilesRendered++
  }

  // ---- WebGL2 / Three.js path ----

  private _ensureThreeMaterial(tile: TerrainTile): void {
    if (!tile.mesh) return

    // Замінюємо дефолтний матеріал на наш elevation shader
    if (tile.mesh.material !== this._material) {
      tile.mesh.material = this._material
    }
  }

  // ---- Texture management ----

  /**
   * Завантажити текстури терейну з URL.
   * Після завантаження — оновити WebGPU bind group.
   */
  async loadTextures(urls: {
    grass?:     string
    rock?:      string
    snow?:      string
    sand?:      string
    normalMap?: string
  }): Promise<void> {
    if (!this._isWebGPU || !this._gpuPipeline) return

    const views: Record<string, GPUTextureView> = {}

    for (const [key, url] of Object.entries(urls)) {
      if (!url) continue
      try {
        const view = await _loadTextureFromUrl(this._opts.device!, url)
        views[key] = view
      } catch (err) {
        console.warn(`[TerrainRenderer] Не вдалося завантажити текстуру ${key}:`, err)
      }
    }

    if (Object.keys(views).length > 0) {
      this._defaultTextureBG = this._gpuPipeline.createTextureBindGroup(views as any)
      console.log(`[TerrainRenderer] Завантажено ${Object.keys(views).length} текстур`)
    }
  }

  // ---- Налаштування ----

  setTimeOfDay(hours: number): void {
    this._timeOfDay = (hours % 24) / 24
  }

  setWireframe(enabled: boolean): void {
    if (this._material instanceof THREE.MeshStandardMaterial ||
        this._material instanceof THREE.ShaderMaterial) {
      (this._material as THREE.MeshStandardMaterial).wireframe = enabled
    }
  }

  // ---- Getters ----

  get stats():     RenderStats { return { ...this._stats } }
  get isWebGPU():  boolean     { return this._isWebGPU }
  get material():  THREE.Material { return this._material }

  // ---- Dispose ----

  dispose(): void {
    this._gpuPipeline?.dispose()
    this._tileBuffers.clear()
    this._material.dispose()
  }

  // ---- Private helpers ----

  private _buildFogParams() {
    const f = this._opts.fog ?? {}
    return {
      color:   f.color   ?? [0.8, 0.9, 1.0] as [number, number, number],
      density: f.density ?? 0.000015,
      start:   f.start   ?? 10_000,
      end:     f.end     ?? 500_000,
      height:  f.height  ?? 5_000,
    }
  }
}

// ----------------------------------------------------------------
// УТИЛІТИ
// ----------------------------------------------------------------

function _uploadBuffer(
  device:  GPUDevice,
  data:    Float32Array | Uint32Array,
  usage:   number,
  label:   string,
): GPUBuffer {
  const buf = device.createBuffer({
    label,
    size:  Math.max(data.byteLength, 4),
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  })

  if (data instanceof Float32Array) {
    new Float32Array(buf.getMappedRange()).set(data)
  } else {
    new Uint32Array(buf.getMappedRange()).set(data)
  }
  buf.unmap()
  return buf
}

async function _loadTextureFromUrl(
  device: GPUDevice,
  url:    string,
): Promise<GPUTextureView> {
  const img = await createImageBitmap(
    await fetch(url).then(r => r.blob())
  )

  const texture = device.createTexture({
    size:   [img.width, img.height],
    format: 'rgba8unorm',
    usage:  GPUTextureUsage.TEXTURE_BINDING
           | GPUTextureUsage.COPY_DST
           | GPUTextureUsage.RENDER_ATTACHMENT,
    mipLevelCount: Math.floor(Math.log2(Math.max(img.width, img.height))) + 1,
  })

  device.queue.copyExternalImageToTexture(
    { source: img },
    { texture },
    [img.width, img.height],
  )

  return texture.createView()
    }
