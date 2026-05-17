/**
 * Vitest global setup — мок Three.js WebGL context
 */

import { vi } from "vitest"

// Three.js потребує WebGL у браузері — мокуємо для jsdom
Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
  value: vi.fn().mockReturnValue({
    // WebGL2 мок
    getParameter:        vi.fn().mockReturnValue(null),
    getExtension:        vi.fn().mockReturnValue(null),
    createBuffer:        vi.fn().mockReturnValue({}),
    bindBuffer:          vi.fn(),
    bufferData:          vi.fn(),
    createProgram:       vi.fn().mockReturnValue({}),
    createShader:        vi.fn().mockReturnValue({}),
    shaderSource:        vi.fn(),
    compileShader:       vi.fn(),
    attachShader:        vi.fn(),
    linkProgram:         vi.fn(),
    getProgramParameter: vi.fn().mockReturnValue(true),
    getShaderParameter:  vi.fn().mockReturnValue(true),
    useProgram:          vi.fn(),
    viewport:            vi.fn(),
    clear:               vi.fn(),
    drawElements:        vi.fn(),
    enable:              vi.fn(),
    disable:             vi.fn(),
    depthFunc:           vi.fn(),
    blendFunc:           vi.fn(),
    createTexture:       vi.fn().mockReturnValue({}),
    bindTexture:         vi.fn(),
    texImage2D:          vi.fn(),
    texParameteri:       vi.fn(),
    createVertexArray:   vi.fn().mockReturnValue({}),
    bindVertexArray:     vi.fn(),
    vertexAttribPointer: vi.fn(),
    enableVertexAttribArray: vi.fn(),
    createFramebuffer:   vi.fn().mockReturnValue({}),
    bindFramebuffer:     vi.fn(),
    createRenderbuffer:  vi.fn().mockReturnValue({}),
    bindRenderbuffer:    vi.fn(),
    renderbufferStorage: vi.fn(),
    framebufferRenderbuffer: vi.fn(),
    checkFramebufferStatus: vi.fn().mockReturnValue(36053), // FRAMEBUFFER_COMPLETE
  }),
})

// Мок performance.now() для стабільних тестів
vi.spyOn(performance, "now").mockReturnValue(1000)

// Мок crypto.randomUUID()
Object.defineProperty(global, "crypto", {
  value: { randomUUID: vi.fn().mockReturnValue("test-uuid-1234") },
})
