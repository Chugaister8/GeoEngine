/**
 * GeoEngine — useGeoEngine Hook
 * React hook для управління GeoRenderer lifecycle.
 *
 * Відповідає за:
 * - Ініціалізацію рендерера при монтуванні
 * - Cleanup при анмаунтуванні
 * - Реактивний стан (fps, tiles, помилки)
 * - Команди камери (flyTo, setCameraPosition)
 * - WebSocket статус
 */

"use client"

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react"
import type { RefObject } from "react"

import type { GeoRendererOptions, RenderStats, RendererState } from "@geoengine/core-js"

// ----------------------------------------------------------------
// ТИПИ
// ----------------------------------------------------------------

export interface UseGeoEngineOptions {
  /** WebSocket URL до Python сервера */
  serverUrl?:     string

  /** Початкова позиція камери */
  initialCamera?: {
    lat:      number
    lon:      number
    altitude: number
  }

  /** Origin для ENU координат */
  origin?: {
    lat: number
    lon: number
  }

  /** WebGPU / WebGL2 / auto */
  renderer?: "webgpu" | "webgl2" | "auto"

  /** Увімкнути debug overlay */
  debug?: boolean

  /** Callback при готовності рендерера */
  onReady?: () => void

  /** Callback при помилці */
  onError?: (error: Error) => void
}

export interface UseGeoEngineReturn {
  /** Ref для canvas елементу */
  canvasRef:  RefObject<HTMLCanvasElement>

  /** Поточний стан рендерера */
  state:      RendererState

  /** Чи готовий рендерер */
  isReady:    boolean

  /** Поточна статистика рендерингу */
  stats:      RenderStats | null

  /** Поточна помилка */
  error:      Error | null

  /** Переміститися до координат */
  flyTo:      (lat: number, lon: number, altitude: number, duration?: number) => void

  /** Встановити позицію камери миттєво */
  setCamera:  (lat: number, lon: number, altitude: number) => void

  /** Пауза / продовження рендерингу */
  togglePause: () => void

  /** Зробити скріншот */
  screenshot: () => string | null

  /** Чи підключений WebSocket */
  wsConnected: boolean
}

// ----------------------------------------------------------------
// HOOK
// ----------------------------------------------------------------

export function useGeoEngine(
  options: UseGeoEngineOptions = {},
): UseGeoEngineReturn {
  const {
    serverUrl     = "ws://localhost:8000/ws",
    initialCamera = { lat: 48.25, lon: 23.5, altitude: 5000 },
    origin,
    renderer      = "auto",
    debug         = false,
    onReady,
    onError,
  } = options

  const canvasRef   = useRef<HTMLCanvasElement>(null)
  const engineRef   = useRef<any>(null)       // GeoRenderer instance
  const pausedRef   = useRef(false)

  const [state,       setState]       = useState<RendererState>("idle")
  const [stats,       setStats]       = useState<RenderStats | null>(null)
  const [error,       setError]       = useState<Error | null>(null)
  const [wsConnected, setWsConnected] = useState(false)

  // ---- Ініціалізація ----

  useEffect(() => {
    if (!canvasRef.current) return

    let cancelled = false

    const init = async (): Promise<void> => {
      try {
        // Динамічний імпорт щоб не ламати SSR
        const { GeoRenderer } = await import("@geoengine/core-js")

        if (cancelled) return

        const engineOptions: GeoRendererOptions = {
          canvas:        canvasRef.current!,
          renderer,
          serverUrl,
          initialCamera,
          originLat:     origin?.lat ?? initialCamera.lat,
          originLon:     origin?.lon ?? initialCamera.lon,
          debug,
        }

        const engine = new GeoRenderer(engineOptions)
        engineRef.current = engine

        setState("initializing")
        await engine.init()

        if (cancelled) {
          engine.dispose()
          return
        }

        engine.start()
        setState("running")
        setWsConnected(true)
        onReady?.()

        // Оновлення статистики кожні 500мс
        const statsInterval = setInterval(() => {
          if (engineRef.current && !pausedRef.current) {
            setStats({ ...engineRef.current.stats })
          }
        }, 500)

        // Cleanup
        return () => {
          clearInterval(statsInterval)
        }

      } catch (err) {
        if (cancelled) return
        const e = err instanceof Error ? err : new Error(String(err))
        setError(e)
        setState("error")
        onError?.(e)
        console.error("[useGeoEngine] init error:", e)
      }
    }

    const cleanup = init()

    return () => {
      cancelled = true
      cleanup.then(cleanupFn => cleanupFn?.())
      if (engineRef.current) {
        engineRef.current.dispose()
        engineRef.current = null
      }
      setState("idle")
      setWsConnected(false)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])  // Навмисно порожній масив — ініціалізуємо один раз

  // ---- Команди ----

  const flyTo = useCallback((
    lat:      number,
    lon:      number,
    altitude: number,
    duration  = 2000,
  ): void => {
    engineRef.current?.flyTo({ lat, lon, altitude, duration })
  }, [])

  const setCamera = useCallback((
    lat: number,
    lon: number,
    altitude: number,
  ): void => {
    engineRef.current?.setCameraPosition(lat, lon, altitude)
  }, [])

  const togglePause = useCallback((): void => {
    const engine = engineRef.current
    if (!engine) return

    if (pausedRef.current) {
      engine.start()
      pausedRef.current = false
      setState("running")
    } else {
      engine.stop()
      pausedRef.current = true
      setState("paused")
    }
  }, [])

  const screenshot = useCallback((): string | null => {
    const canvas = canvasRef.current
    if (!canvas) return null
    return canvas.toDataURL("image/png")
  }, [])

  const isReady = state === "running" || state === "paused"

  return {
    canvasRef,
    state,
    isReady,
    stats,
    error,
    flyTo,
    setCamera,
    togglePause,
    screenshot,
    wsConnected,
  }
}
