/**
 * GeoEngine — GeoCanvas Component
 * Головний 3D вьюпорт.
 *
 * Рендерить WebGPU/WebGL canvas + overlay UI:
 *   - Stats overlay (FPS, трикутники, тайли)
 *   - Координати курсору
 *   - Компас
 *   - Toolbar кнопки
 *   - Error стан
 *   - Loading стан
 */

"use client"

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type MouseEvent,
} from "react"

import { useGeoEngine, type UseGeoEngineOptions } from "../hooks/useGeoEngine"

// ----------------------------------------------------------------
// ТИПИ
// ----------------------------------------------------------------

export interface GeoCanvasProps extends UseGeoEngineOptions {
  className?:   string
  showStats?:   boolean
  showCompass?: boolean
  showCoords?:  boolean
  showToolbar?: boolean
  height?:      string   // CSS height, default "100vh"
}

// ----------------------------------------------------------------
// КОМПОНЕНТ
// ----------------------------------------------------------------

export function GeoCanvas({
  className,
  showStats   = true,
  showCompass = true,
  showCoords  = true,
  showToolbar = true,
  height      = "100vh",
  ...engineOptions
}: GeoCanvasProps) {
  const {
    canvasRef,
    state,
    isReady,
    stats,
    error,
    flyTo,
    togglePause,
    screenshot,
    wsConnected,
  } = useGeoEngine(engineOptions)

  // Координати курсору над canvas
  const [cursorCoords, setCursorCoords] = useState<{
    lat: number; lon: number
  } | null>(null)

  // Обробка руху миші для координат
  const handleMouseMove = useCallback((e: MouseEvent<HTMLCanvasElement>) => {
    // TODO: raycast canvas pixel → lat/lon через GeoRenderer API
    // Поки що — заглушка
    const rect   = e.currentTarget.getBoundingClientRect()
    const relX   = (e.clientX - rect.left) / rect.width
    const relY   = (e.clientY - rect.top)  / rect.height

    if (engineOptions.initialCamera) {
      const { lat, lon } = engineOptions.initialCamera
      setCursorCoords({
        lat: lat + (0.5 - relY) * 0.1,
        lon: lon + (relX - 0.5) * 0.1,
      })
    }
  }, [engineOptions.initialCamera])

  const handleMouseLeave = useCallback(() => {
    setCursorCoords(null)
  }, [])

  // ---- Рендер ----

  return (
    <div
      className={`relative overflow-hidden bg-black ${className ?? ""}`}
      style={{ height }}
    >
      {/* Основний canvas */}
      <canvas
        ref={canvasRef}
        className="block w-full h-full"
        onMouseMove={showCoords ? handleMouseMove : undefined}
        onMouseLeave={showCoords ? handleMouseLeave : undefined}
      />

      {/* Loading overlay */}
      {state === "initializing" && (
        <LoadingOverlay />
      )}

      {/* Error overlay */}
      {state === "error" && error && (
        <ErrorOverlay error={error} />
      )}

      {/* Stats overlay */}
      {showStats && isReady && stats && (
        <StatsOverlay stats={stats} wsConnected={wsConnected} />
      )}

      {/* Compass */}
      {showCompass && isReady && (
        <CompassWidget />
      )}

      {/* Координати курсору */}
      {showCoords && isReady && cursorCoords && (
        <CoordsDisplay coords={cursorCoords} />
      )}

      {/* Toolbar */}
      {showToolbar && isReady && (
        <Toolbar
          onFlyTo={flyTo}
          onTogglePause={togglePause}
          onScreenshot={screenshot}
          isPaused={state === "paused"}
        />
      )}

      {/* WS статус індикатор */}
      <WSStatusDot connected={wsConnected} />
    </div>
  )
}

// ================================================================
// SUB-COMPONENTS
// ================================================================

// ---- Loading ----

function LoadingOverlay() {
  return (
    <div className="
      absolute inset-0 flex flex-col items-center justify-center
      bg-black/80 backdrop-blur-sm z-50
    ">
      <div className="flex flex-col items-center gap-4">
        {/* Spinner */}
        <div className="
          w-12 h-12 border-4 border-white/20
          border-t-white rounded-full animate-spin
        " />
        <div className="text-white/90 text-sm font-mono tracking-wider uppercase">
          Ініціалізація рендерера...
        </div>
        <div className="text-white/40 text-xs font-mono">
          WebGPU · Terrain · OSM
        </div>
      </div>
    </div>
  )
}

// ---- Error ----

function ErrorOverlay({ error }: { error: Error }) {
  return (
    <div className="
      absolute inset-0 flex flex-col items-center justify-center
      bg-black/90 z-50 p-8
    ">
      <div className="
        max-w-md w-full bg-red-950/80 border border-red-500/50
        rounded-lg p-6 backdrop-blur
      ">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-red-400 text-xl">⚠</span>
          <h3 className="text-red-300 font-mono font-bold">
            Помилка рендерера
          </h3>
        </div>
        <p className="text-red-200/80 text-sm font-mono break-all">
          {error.message}
        </p>
        <p className="mt-3 text-white/40 text-xs">
          Перевірте підтримку WebGPU у вашому браузері
          або спробуйте WebGL2 режим.
        </p>
      </div>
    </div>
  )
}

// ---- Stats Overlay ----

interface StatsOverlayProps {
  stats:       import("@geoengine/core-js").RenderStats
  wsConnected: boolean
}

function StatsOverlay({ stats, wsConnected }: StatsOverlayProps) {
  return (
    <div className="
      absolute top-3 left-3 z-10
      font-mono text-xs text-white/80
      bg-black/50 backdrop-blur rounded-lg
      px-3 py-2 space-y-0.5
      border border-white/10
      min-w-[140px]
    ">
      <StatRow label="FPS"     value={stats.fps.toString()} />
      <StatRow label="Frame"   value={`${stats.frameTime}ms`} />
      <StatRow
        label="Tris"
        value={formatNumber(stats.triangles)}
        dimmed={stats.triangles === 0}
      />
      <StatRow label="Tiles"   value={`${stats.visibleTiles}/${stats.totalTiles}`} />
      <StatRow label="Draw"    value={stats.drawCalls.toString()} />
      <div className="border-t border-white/10 pt-0.5 mt-1">
        <StatRow
          label="QT depth"
          value={stats.quadtree.maxDepth.toString()}
        />
        <StatRow
          label="QT nodes"
          value={`${stats.quadtree.leaves}/${stats.quadtree.total}`}
        />
      </div>
    </div>
  )
}

function StatRow({
  label,
  value,
  dimmed = false,
}: {
  label:  string
  value:  string
  dimmed?: boolean
}) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-white/40">{label}</span>
      <span className={dimmed ? "text-white/30" : "text-white/90"}>{value}</span>
    </div>
  )
}

// ---- Compass ----

function CompassWidget() {
  // TODO: підключити до camera.heading
  const heading = 0

  return (
    <div className="
      absolute top-3 right-3 z-10
      w-12 h-12
      bg-black/50 backdrop-blur rounded-full
      border border-white/20
      flex items-center justify-center
      select-none
    ">
      <svg
        viewBox="0 0 40 40"
        className="w-8 h-8"
        style={{ transform: `rotate(${-heading}deg)`, transition: "transform 0.2s" }}
      >
        {/* Північ — червона */}
        <polygon
          points="20,4 17,20 23,20"
          fill="#ef4444"
          opacity="0.9"
        />
        {/* Південь — білий */}
        <polygon
          points="20,36 17,20 23,20"
          fill="white"
          opacity="0.6"
        />
        {/* Центр */}
        <circle cx="20" cy="20" r="2" fill="white" opacity="0.8" />
      </svg>
      <span className="
        absolute bottom-0.5 text-[8px] font-mono
        text-white/60 tracking-widest
      ">
        N
      </span>
    </div>
  )
}

// ---- Coordinates ----

function CoordsDisplay({ coords }: { coords: { lat: number; lon: number } }) {
  return (
    <div className="
      absolute bottom-8 left-1/2 -translate-x-1/2 z-10
      font-mono text-xs text-white/80
      bg-black/50 backdrop-blur rounded
      px-3 py-1.5
      border border-white/10
      pointer-events-none
    ">
      <span className="text-white/40">LAT </span>
      <span>{coords.lat.toFixed(6)}°</span>
      <span className="mx-3 text-white/20">|</span>
      <span className="text-white/40">LON </span>
      <span>{coords.lon.toFixed(6)}°</span>
    </div>
  )
}

// ---- Toolbar ----

interface ToolbarProps {
  onFlyTo:        (lat: number, lon: number, altitude: number) => void
  onTogglePause:  () => void
  onScreenshot:   () => string | null
  isPaused:       boolean
}

function Toolbar({ onFlyTo, onTogglePause, onScreenshot, isPaused }: ToolbarProps) {
  const [showFlyTo, setShowFlyTo] = useState(false)
  const [flyLat,    setFlyLat]    = useState("48.2522")
  const [flyLon,    setFlyLon]    = useState("23.5140")
  const [flyAlt,    setFlyAlt]    = useState("3000")

  const handleFlyTo = useCallback(() => {
    const lat = parseFloat(flyLat)
    const lon = parseFloat(flyLon)
    const alt = parseFloat(flyAlt)
    if (!isNaN(lat) && !isNaN(lon) && !isNaN(alt)) {
      onFlyTo(lat, lon, alt)
      setShowFlyTo(false)
    }
  }, [flyLat, flyLon, flyAlt, onFlyTo])

  const handleScreenshot = useCallback(() => {
    const dataUrl = onScreenshot()
    if (!dataUrl) return
    const a = document.createElement("a")
    a.href = dataUrl
    a.download = `geoengine-${Date.now()}.png`
    a.click()
  }, [onScreenshot])

  return (
    <>
      {/* Toolbar кнопки */}
      <div className="
        absolute bottom-4 right-4 z-10
        flex flex-col gap-2
      ">
        <ToolbarButton
          onClick={() => setShowFlyTo(v => !v)}
          title="Перелетіти до координат"
          active={showFlyTo}
        >
          ✈
        </ToolbarButton>

        <ToolbarButton
          onClick={onTogglePause}
          title={isPaused ? "Відновити" : "Пауза"}
          active={isPaused}
        >
          {isPaused ? "▶" : "⏸"}
        </ToolbarButton>

        <ToolbarButton
          onClick={handleScreenshot}
          title="Скріншот"
        >
          📷
        </ToolbarButton>
      </div>

      {/* FlyTo панель */}
      {showFlyTo && (
        <div className="
          absolute bottom-4 right-16 z-10
          bg-black/80 backdrop-blur border border-white/20
          rounded-lg p-4 space-y-3
          min-w-[220px]
        ">
          <div className="text-white/60 text-xs font-mono uppercase tracking-wider mb-2">
            Перелет до координат
          </div>

          <CoordInput
            label="Широта"
            value={flyLat}
            onChange={setFlyLat}
            placeholder="48.2522"
          />
          <CoordInput
            label="Довгота"
            value={flyLon}
            onChange={setFlyLon}
            placeholder="23.5140"
          />
          <CoordInput
            label="Висота (м)"
            value={flyAlt}
            onChange={setFlyAlt}
            placeholder="3000"
          />

          <div className="flex gap-2 pt-1">
            <button
              onClick={handleFlyTo}
              className="
                flex-1 bg-blue-600 hover:bg-blue-500
                text-white text-xs font-mono
                rounded px-3 py-1.5
                transition-colors
              "
            >
              Летіти ✈
            </button>
            <button
              onClick={() => setShowFlyTo(false)}
              className="
                px-3 py-1.5 text-xs font-mono
                text-white/60 hover:text-white
                transition-colors
              "
            >
              ✕
            </button>
          </div>

          {/* Швидкі локації */}
          <div className="border-t border-white/10 pt-2">
            <div className="text-white/30 text-xs mb-1.5">Швидкий перехід</div>
            <div className="grid grid-cols-2 gap-1">
              {QUICK_LOCATIONS.map(loc => (
                <button
                  key={loc.name}
                  onClick={() => {
                    onFlyTo(loc.lat, loc.lon, loc.alt)
                    setShowFlyTo(false)
                  }}
                  className="
                    text-left text-xs font-mono
                    text-white/50 hover:text-white/90
                    py-0.5 px-1 rounded
                    hover:bg-white/10
                    transition-colors truncate
                  "
                >
                  {loc.name}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  )
}

function ToolbarButton({
  children,
  onClick,
  title,
  active = false,
}: {
  children: React.ReactNode
  onClick:  () => void
  title?:   string
  active?:  boolean
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={`
        w-10 h-10 rounded-lg text-base
        border transition-all duration-150
        flex items-center justify-center
        backdrop-blur font-emoji
        ${active
          ? "bg-blue-600/80 border-blue-400/60 text-white"
          : "bg-black/60 border-white/20 text-white/70 hover:bg-black/80 hover:text-white hover:border-white/40"
        }
      `}
    >
      {children}
    </button>
  )
}

function CoordInput({
  label,
  value,
  onChange,
  placeholder,
}: {
  label:       string
  value:       string
  onChange:    (v: string) => void
  placeholder: string
}) {
  return (
    <div className="space-y-1">
      <label className="text-white/40 text-xs font-mono">{label}</label>
      <input
        type="number"
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        step="0.0001"
        className="
          w-full bg-white/10 border border-white/20
          rounded px-2 py-1
          text-white text-xs font-mono
          placeholder-white/20
          focus:outline-none focus:border-blue-400/60
          transition-colors
        "
      />
    </div>
  )
}

// ---- WS Status Dot ----

function WSStatusDot({ connected }: { connected: boolean }) {
  return (
    <div
      className="absolute top-3 right-16 z-10"
      title={connected ? "Сервер підключений" : "Сервер відключений"}
    >
      <div className={`
        w-2 h-2 rounded-full
        ${connected
          ? "bg-green-400 shadow-[0_0_6px_rgba(74,222,128,0.8)]"
          : "bg-red-500/60"
        }
      `} />
    </div>
  )
}

// ----------------------------------------------------------------
// УТИЛІТИ
// ----------------------------------------------------------------

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000)     return `${(n / 1_000).toFixed(0)}K`
  return n.toString()
}

const QUICK_LOCATIONS = [
  { name: "Карпати",  lat: 48.15, lon: 24.50, alt: 8000  },
  { name: "Київ",     lat: 50.45, lon: 30.52, alt: 5000  },
  { name: "Одеса",    lat: 46.48, lon: 30.72, alt: 4000  },
  { name: "Крим",     lat: 44.95, lon: 34.10, alt: 10000 },
  { name: "Говерла",  lat: 48.16, lon: 24.50, alt: 3000  },
  { name: "Дніпро",   lat: 48.46, lon: 35.04, alt: 4000  },
] as const
