/**
 * GeoEngine — WebGPU Pipeline Manager
 * Управляє всім WebGPU render pipeline:
 *   - Canvas context та swap chain
 *   - MSAA resolve targets
 *   - Depth buffer
 *   - Render pass descriptor
 *   - Frame timing
 *
 * Один Pipeline на весь GeoRenderer.
 * TerrainRenderer, AtmosphereRenderer, WaterRenderer
 * записують свої команди в shared render pass.
 */

import * as THREE from 'three'

// ----------------------------------------------------------------
// PIPELINE OPTIONS
// ----------------------------------------------------------------

export interface PipelineOptions {
  canvas:      HTMLCanvasElement
  msaaSamples?: number   // 1 або 4, default 4
  hdr?:        boolean   // HDR output, default false
}

// ----------------------------------------------------------------
// WEBGPU PIPELINE
// ----------------------------------------------------------------

export class WebGPUPipeline {
  /**
   * Ініціалізує повний WebGPU context.
   *
   * Lifecycle:
   *   init() → beginFrame() → [render passes] → endFrame() → dispose()
   */

  private _device!:      GPUDevice
  private _context!:     GPUCanvasContext
  private _format!:      GPUTextureFormat
  private _ready:        boolean = false

  // MSAA
  private _msaaTexture:  GPUTexture | null = null
  private _msaaView:     GPUTextureView | null = null
  private _msaaSamples:  number = 4

  // Depth
  private _depthTexture: GPUTexture | null = null
  private _depthView:    GPUTextureView | null = null

  // Frame
  private _currentEncoder:  GPUCommandEncoder | null = null
  private _currentPass:     GPURenderPassEncoder | null = null
  private _frameStart:      number = 0

  // Stats
  private _frameCount = 0
  private _lastFPS    = 0
  private _fpsTime    = 0

  constructor(private readonly _opts: PipelineOptions) {
    this._msaaSamples = _opts.msaaSamples ?? 4
  }

  // ---- Ініціалізація ----

  async init(): Promise<boolean> {
    if (!('gpu' in navigator)) {
      console.warn('[WebGPUPipeline] WebGPU не підтримується')
      return false
    }

    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      })
      if (!adapter) {
        console.warn('[WebGPUPipeline] GPU adapter не знайдено')
        return false
      }

      this._device = await adapter.requestDevice({
        requiredFeatures: [],
        requiredLimits: {
          maxBindGroups:            4,
          maxUniformBufferBindingSize: 65536,
        },
      })

      this._device.lost.then(info => {
        console.error('[WebGPUPipeline] Device lost:', info.message)
        this._ready = false
      })

      this._context = this._opts.canvas.getContext('webgpu') as GPUCanvasContext
      if (!this._context) {
        console.warn('[WebGPUPipeline] Не вдалося отримати webgpu context')
        return false
      }

      this._format = navigator.gpu.getPreferredCanvasFormat()
      this._context.configure({
        device:     this._device,
        format:     this._format,
        alphaMode:  'premultiplied',
      })

      this._createRenderTargets()
      this._ready = true

      console.log('[WebGPUPipeline] ініціалізовано', {
        adapter:  adapter.info?.architecture ?? 'unknown',
        format:   this._format,
        msaa:     this._msaaSamples,
      })

      return true

    } catch (err) {
      console.warn('[WebGPUPipeline] init failed:', err)
      return false
    }
  }

  // ---- Frame lifecycle ----

  /**
   * Почати новий кадр.
   * Повертає render pass encoder або null якщо pipeline не готовий.
   */
  beginFrame(): GPURenderPassEncoder | null {
    if (!this._ready) return null

    this._frameStart   = performance.now()
    this._currentEncoder = this._device.createCommandEncoder({
      label: `frame-${this._frameCount}`,
    })

    const currentTexture = this._context.getCurrentTexture()
    const resolveTarget  = currentTexture.createView()

    const colorAttachment: GPURenderPassColorAttachment = this._msaaSamples > 1
      ? {
          view:        this._msaaView!,
          resolveTarget,
          clearValue:  { r: 0.53, g: 0.81, b: 1.0, a: 1.0 },  // sky blue
          loadOp:      'clear',
          storeOp:     'discard',
        }
      : {
          view:        resolveTarget,
          clearValue:  { r: 0.53, g: 0.81, b: 1.0, a: 1.0 },
          loadOp:      'clear',
          storeOp:     'store',
        }

    this._currentPass = this._currentEncoder.beginRenderPass({
      label: 'main-pass',
      colorAttachments: [colorAttachment],
      depthStencilAttachment: {
        view:              this._depthView!,
        depthClearValue:   1.0,
        depthLoadOp:       'clear',
        depthStoreOp:      'discard',
      },
    })

    return this._currentPass
  }

  /**
   * Завершити кадр та відправити на GPU.
   */
  endFrame(): void {
    if (!this._currentPass || !this._currentEncoder) return

    this._currentPass.end()
    this._device.queue.submit([this._currentEncoder.finish()])

    this._currentPass    = null
    this._currentEncoder = null
    this._frameCount++

    // FPS calculation
    const now = performance.now()
    if (now - this._fpsTime > 1000) {
      this._lastFPS  = this._frameCount
      this._frameCount = 0
      this._fpsTime    = now
    }
  }

  // ---- Resize ----

  resize(width: number, height: number): void {
    if (!this._ready) return

    this._opts.canvas.width  = width
    this._opts.canvas.height = height

    // Пересоздаємо render targets під новий розмір
    this._msaaTexture?.destroy()
    this._depthTexture?.destroy()
    this._createRenderTargets()
  }

  // ---- Getters ----

  get device():   GPUDevice  { return this._device }
  get format():   GPUTextureFormat { return this._format }
  get isReady():  boolean    { return this._ready }
  get fps():      number     { return this._lastFPS }
  get frameTimeMs(): number  { return performance.now() - this._frameStart }

  // ---- Dispose ----

  dispose(): void {
    this._msaaTexture?.destroy()
    this._depthTexture?.destroy()
    this._device?.destroy()
    this._ready = false
  }

  // ---- Приватні методи ----

  private _createRenderTargets(): void {
    const { canvas } = this._opts
    const w = canvas.width  || 1
    const h = canvas.height || 1

    // MSAA texture
    if (this._msaaSamples > 1) {
      this._msaaTexture = this._device.createTexture({
        label:       'msaa-color',
        size:        [w, h],
        sampleCount: this._msaaSamples,
        format:      this._format,
        usage:       GPUTextureUsage.RENDER_ATTACHMENT,
      })
      this._msaaView = this._msaaTexture.createView()
    }

    // Depth texture
    this._depthTexture = this._device.createTexture({
      label:       'depth',
      size:        [w, h],
      sampleCount: this._msaaSamples,
      format:      'depth24plus',
      usage:       GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this._depthView = this._depthTexture.createView()
  }
}

// ----------------------------------------------------------------
// THREE.JS PIPELINE (WebGL2 fallback)
// ----------------------------------------------------------------

export class ThreeJSPipeline {
  /**
   * Обгортка навколо THREE.WebGLRenderer.
   * Використовується як fallback коли WebGPU недоступний.
   *
   * Надає той самий API що WebGPUPipeline де можливо.
   */

  private _renderer: THREE.WebGLRenderer
  private _scene:    THREE.Scene
  private _camera:   THREE.Camera

  constructor(
    renderer: THREE.WebGLRenderer,
    scene:    THREE.Scene,
    camera:   THREE.Camera,
  ) {
    this._renderer = renderer
    this._scene    = scene
    this._camera   = camera
  }

  // Заглушки для API сумісності
  beginFrame(): null { return null }

  endFrame(): void {
    this._renderer.render(this._scene, this._camera)
  }

  resize(w: number, h: number): void {
    this._renderer.setSize(w, h)
    if (this._camera instanceof THREE.PerspectiveCamera) {
      this._camera.aspect = w / h
      this._camera.updateProjectionMatrix()
    }
  }

  get device(): null    { return null }
  get format(): string  { return 'webgl2' }
  get isReady(): boolean { return true }
  get fps(): number     { return this._renderer.info.render.frame ?? 0 }

  dispose(): void { this._renderer.dispose() }
      }
