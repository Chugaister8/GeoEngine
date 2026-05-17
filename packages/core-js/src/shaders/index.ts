/**
 * GeoEngine — Shader Registry
 * Централізований реєстр WGSL шейдерів.
 *
 * Імпортуємо як raw strings через Vite/webpack raw-loader.
 * next.config.ts вже налаштований для *.wgsl → asset/source.
 */

// Raw WGSL imports (через webpack asset/source)
// @ts-ignore — wgsl файли не мають TypeScript типів
import terrainWGSL    from "./terrain.wgsl"
// @ts-ignore
import atmosphereWGSL from "./atmosphere.wgsl"
// @ts-ignore
import waterWGSL      from "./water.wgsl"
// @ts-ignore
import buildingsWGSL  from "./buildings.wgsl"

export {
  terrainWGSL,
  atmosphereWGSL,
  waterWGSL,
  buildingsWGSL,
}

// ----------------------------------------------------------------
// SHADER COMPILER (WebGPU)
// ----------------------------------------------------------------

export interface CompiledShader {
  vertex:   GPUShaderModule
  fragment: GPUShaderModule
}

/**
 * Скомпілювати WGSL шейдер у GPUShaderModule.
 * Додає label для debug та перехоплює помилки компіляції.
 */
export async function compileShader(
  device:    GPUDevice,
  code:      string,
  label:     string,
): Promise<GPUShaderModule> {
  const module = device.createShaderModule({
    label,
    code,
  })

  // WebGPU async compilation info (Chrome 113+)
  if ("getCompilationInfo" in module) {
    const info = await (module as any).getCompilationInfo()
    for (const msg of info.messages) {
      if (msg.type === "error") {
        throw new Error(
          `[${label}] WGSL compile error at ${msg.lineNum}:${msg.linePos}\n${msg.message}`
        )
      }
      if (msg.type === "warning") {
        console.warn(`[${label}] WGSL warning:`, msg.message)
      }
    }
  }

  return module
}

/**
 * Скомпілювати terrain шейдер.
 */
export async function compileTerrainShader(
  device: GPUDevice,
): Promise<GPUShaderModule> {
  return compileShader(device, terrainWGSL as string, "GeoEngine/terrain")
}

/**
 * Скомпілювати atmosphere шейдер.
 */
export async function compileAtmosphereShader(
  device: GPUDevice,
): Promise<GPUShaderModule> {
  return compileShader(device, atmosphereWGSL as string, "GeoEngine/atmosphere")
}

/**
 * Скомпілювати water шейдер.
 */
export async function compileWaterShader(
  device: GPUDevice,
): Promise<GPUShaderModule> {
  return compileShader(device, waterWGSL as string, "GeoEngine/water")
}

/**
 * Скомпілювати buildings шейдер.
 */
export async function compileBuildingsShader(
  device: GPUDevice,
): Promise<GPUShaderModule> {
  return compileShader(device, buildingsWGSL as string, "GeoEngine/buildings")
}

// ----------------------------------------------------------------
// UNIFORM BUFFER HELPERS
// ----------------------------------------------------------------

/**
 * Створити GPUBuffer для uniform даних.
 */
export function createUniformBuffer(
  device: GPUDevice,
  size:   number,
  label?: string,
): GPUBuffer {
  return device.createBuffer({
    label,
    size:  Math.max(size, 16),   // мінімум 16 байт (WebGPU вимога)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
}

/**
 * Оновити uniform buffer з Float32Array.
 */
export function updateUniformBuffer(
  device:  GPUDevice,
  buffer:  GPUBuffer,
  data:    Float32Array,
  offset = 0,
): void {
  device.queue.writeBuffer(buffer, offset, data)
}

/**
 * Camera Uniforms → Float32Array (для GPU).
 * Порядок: view(16), projection(16), view_proj(16), position(3), near(1), far(1), pad(3)
 */
export function packCameraUniforms(
  view:       Float32Array,   // mat4
  projection: Float32Array,   // mat4
  viewProj:   Float32Array,   // mat4
  position:   [number, number, number],
  near:       number,
  far:        number,
): Float32Array {
  const data = new Float32Array(16 + 16 + 16 + 4 + 4)
  data.set(view,       0)
  data.set(projection, 16)
  data.set(viewProj,   32)
  data.set(position,   48)
  data[51] = near
  data[52] = far
  return data
}

/**
 * Terrain Uniforms → Float32Array.
 */
export function packTerrainUniforms(params: {
  minElevation:  number
  maxElevation:  number
  lodLevel:      number
  morphFactor:   number
  scale:         number
  uvScale:       number
  seaLevel:      number
  grassEnd:      number
  rockStart:     number
  snowStart:     number
  slopeRockDeg:  number
  slopeSNowDeg:  number
}): Float32Array {
  return new Float32Array([
    params.minElevation,
    params.maxElevation,
    params.lodLevel,
    params.morphFactor,
    params.scale,
    params.uvScale,
    params.seaLevel,
    params.grassEnd,
    params.rockStart,
    params.snowStart,
    params.slopeRockDeg,
    params.slopeSNowDeg,
    0, 0,  // padding
  ])
}

/**
 * Fog Uniforms → Float32Array.
 */
export function packFogUniforms(params: {
  color:   [number, number, number]
  density: number
  start:   number
  end:     number
  height:  number
}): Float32Array {
  return new Float32Array([
    ...params.color,
    params.density,
    params.start,
    params.end,
    params.height,
    0,  // padding
  ])
}
