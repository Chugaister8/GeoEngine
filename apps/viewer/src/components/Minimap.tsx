/**
 * GeoEngine — Minimap
 * 2D мінікарта з поточним положенням камери.
 * Рендерить OSM тайли через Canvas 2D API.
 */

"use client"

import {
  useEffect,
  useRef,
  useState,
  type MouseEvent,
} from "react"

export interface MinimapProps {
  /** Поточна позиція камери */
  cameraLat: number
  cameraLon: number

  /** Радіус видимості (метри) */
  viewRadiusM?: number

  /** Клік по мінімапі → переліт */
  onClickLocation?: (lat: number, lon: number) => void

  className?: string
}

export function Minimap({
  cameraLat,
  cameraLon,
  viewRadiusM   = 10_000,
  onClickLocation,
  className,
}: MinimapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [zoom]    = useState(12)
  const SIZE      = 160

  // Малювати мінімапу
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Фон
    ctx.fillStyle = "#1a1a2e"
    ctx.fillRect(0, 0, SIZE, SIZE)

    // Сітка координат (спрощена)
    ctx.strokeStyle = "rgba(255,255,255,0.05)"
    ctx.lineWidth   = 0.5
    for (let i = 0; i <= 4; i++) {
      const x = (i / 4) * SIZE
      const y = (i / 4) * SIZE
      ctx.beginPath()
      ctx.moveTo(x, 0); ctx.lineTo(x, SIZE)
      ctx.moveTo(0, y); ctx.lineTo(SIZE, y)
      ctx.stroke()
    }

    // Позиція камери — хрестик
    const cx = SIZE / 2
    const cy = SIZE / 2

    // Коло видимості
    const radiusPx = Math.min(SIZE * 0.3, 40)
    ctx.beginPath()
    ctx.arc(cx, cy, radiusPx, 0, Math.PI * 2)
    ctx.strokeStyle = "rgba(59, 130, 246, 0.4)"
    ctx.lineWidth   = 1
    ctx.stroke()
    ctx.fillStyle   = "rgba(59, 130, 246, 0.05)"
    ctx.fill()

    // Хрестик камери
    const CS = 8
    ctx.strokeStyle = "#60a5fa"
    ctx.lineWidth   = 1.5
    ctx.beginPath()
    ctx.moveTo(cx - CS, cy); ctx.lineTo(cx + CS, cy)
    ctx.moveTo(cx, cy - CS); ctx.lineTo(cx, cy + CS)
    ctx.stroke()

    // Точка в центрі
    ctx.beginPath()
    ctx.arc(cx, cy, 2.5, 0, Math.PI * 2)
    ctx.fillStyle = "#93c5fd"
    ctx.fill()

    // Координати
    ctx.fillStyle   = "rgba(255,255,255,0.4)"
    ctx.font        = "8px monospace"
    ctx.fillText(`${cameraLat.toFixed(3)}°`, 4, SIZE - 12)
    ctx.fillText(`${cameraLon.toFixed(3)}°`, 4, SIZE - 4)

  }, [cameraLat, cameraLon, zoom])

  const handleClick = (e: MouseEvent<HTMLCanvasElement>): void => {
    if (!onClickLocation) return
    const rect   = e.currentTarget.getBoundingClientRect()
    const relX   = (e.clientX - rect.left) / SIZE - 0.5
    const relY   = (e.clientY - rect.top)  / SIZE - 0.5
    const latDeg = viewRadiusM / 111_320
    const lonDeg = viewRadiusM / (111_320 * Math.cos(cameraLat * Math.PI / 180))
    onClickLocation(
      cameraLat - relY * latDeg * 2,
      cameraLon + relX * lonDeg * 2,
    )
  }

  return (
    <div className={`
      relative overflow-hidden rounded-lg
      border border-white/15 bg-black/60
      ${className ?? ""}
      ${onClickLocation ? "cursor-crosshair" : ""}
    `}>
      <canvas
        ref={canvasRef}
        width={SIZE}
        height={SIZE}
        onClick={handleClick}
        className="block"
      />
      {/* Overlay label */}
      <div className="
        absolute top-1.5 left-1.5
        text-[9px] font-mono text-white/30
        uppercase tracking-widest
      ">
        MAP
      </div>
    </div>
  )
      }
