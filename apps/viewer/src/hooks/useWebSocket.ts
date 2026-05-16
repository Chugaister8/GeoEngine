/**
 * GeoEngine — useWebSocket Hook
 * Низькорівневий WebSocket hook з автоматичним перепідключенням.
 *
 * Відокремлений від useGeoEngine щоб дозволити
 * ручне надсилання команд на сервер (аналіз, підписки).
 */

"use client"

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react"

// ----------------------------------------------------------------
// ТИПИ
// ----------------------------------------------------------------

export type WSStatus = "connecting" | "connected" | "disconnected" | "error"

export interface UseWebSocketOptions {
  url:            string
  onMessage?:     (data: unknown) => void
  onConnect?:     () => void
  onDisconnect?:  () => void
  reconnectDelay?: number   // мс, default 3000
  maxRetries?:    number    // -1 = нескінченно
}

export interface UseWebSocketReturn {
  status:   WSStatus
  send:     (message: unknown) => boolean
  connect:  () => void
  disconnect: () => void
  latencyMs: number | null
}

// ----------------------------------------------------------------
// HOOK
// ----------------------------------------------------------------

export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
  const {
    url,
    onMessage,
    onConnect,
    onDisconnect,
    reconnectDelay = 3000,
    maxRetries     = -1,
  } = options

  const wsRef        = useRef<WebSocket | null>(null)
  const retriesRef   = useRef(0)
  const mountedRef   = useRef(true)
  const pingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pingStartRef = useRef<number>(0)

  const [status,    setStatus]    = useState<WSStatus>("disconnected")
  const [latencyMs, setLatencyMs] = useState<number | null>(null)

  // ---- Connect ----

  const connect = useCallback((): void => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    if (!mountedRef.current) return

    setStatus("connecting")

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = (): void => {
      if (!mountedRef.current) { ws.close(); return }
      retriesRef.current = 0
      setStatus("connected")
      onConnect?.()

      // Ping кожні 30 секунд для вимірювання latency
      pingTimerRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          pingStartRef.current = performance.now()
          ws.send(JSON.stringify({
            type: "ping",
            id: crypto.randomUUID(),
            timestamp: Date.now(),
            payload: {},
          }))
        }
      }, 30_000)
    }

    ws.onmessage = (event: MessageEvent): void => {
      let data: unknown
      try {
        data = JSON.parse(event.data as string)
      } catch {
        data = event.data
      }

      // Обробка pong для latency
      if (
        typeof data === "object" && data !== null
        && (data as any).type === "pong"
        && pingStartRef.current > 0
      ) {
        setLatencyMs(Math.round(performance.now() - pingStartRef.current))
        return
      }

      onMessage?.(data)
    }

    ws.onerror = (): void => {
      if (!mountedRef.current) return
      setStatus("error")
    }

    ws.onclose = (): void => {
      if (!mountedRef.current) return
      if (pingTimerRef.current) {
        clearInterval(pingTimerRef.current)
        pingTimerRef.current = null
      }
      setStatus("disconnected")
      onDisconnect?.()

      // Автоперепідключення
      const shouldRetry = maxRetries === -1 || retriesRef.current < maxRetries
      if (shouldRetry && mountedRef.current) {
        retriesRef.current++
        setTimeout(() => {
          if (mountedRef.current) connect()
        }, reconnectDelay * Math.min(retriesRef.current, 5))  // exponential backoff cap
      }
    }
  }, [url, onMessage, onConnect, onDisconnect, reconnectDelay, maxRetries])

  // ---- Disconnect ----

  const disconnect = useCallback((): void => {
    if (pingTimerRef.current) {
      clearInterval(pingTimerRef.current)
      pingTimerRef.current = null
    }
    wsRef.current?.close()
    wsRef.current = null
    setStatus("disconnected")
  }, [])

  // ---- Send ----

  const send = useCallback((message: unknown): boolean => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return false
    try {
      wsRef.current.send(JSON.stringify(message))
      return true
    } catch {
      return false
    }
  }, [])

  // ---- Lifecycle ----

  useEffect(() => {
    mountedRef.current = true
    connect()

    return () => {
      mountedRef.current = false
      disconnect()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url])

  return { status, send, connect, disconnect, latencyMs }
        }
