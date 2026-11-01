import { describe, expect, it, vi } from 'vitest'

import {
  activateCanvas,
  createCanvas,
  prefetchCanvasPages,
  previewCanvasOpacity,
  showCanvasPage,
  type Canvas,
} from '@koharu/bridge/canvas'

const commands = vi.hoisted(() => ({
  prepareCanvasPage: vi.fn(),
  getCanvasPageManifest: vi.fn(),
  getCanvasPageResource: vi.fn(),
}))

const wasm = vi.hoisted(() => ({
  default: vi.fn(async () => undefined),
  createCanvas: vi.fn(),
}))

vi.mock('@koharu/bridge/protocol', () => ({ commands }))
vi.mock('@/tests/mocks/koharu_canvas', () => wasm)

const manifest = new Uint8Array([1, 2, 3])

describe('canvas runtime', () => {
  it('routes inspector previews only to the active canvas', () => {
    const previewOpacity = vi.fn()
    const canvas = { previewOpacity } as unknown as Canvas
    const deactivate = activateCanvas(canvas)

    previewCanvasOpacity('layer', 0.5)
    deactivate()
    previewCanvasOpacity('layer', null)

    expect(previewOpacity).toHaveBeenCalledOnce()
    expect(previewOpacity).toHaveBeenCalledWith('layer', 0.5)
  })

  it('shows a cached page immediately on the active canvas', () => {
    const activatePage = vi.fn(() => true)
    const canvas = { activatePage } as unknown as Canvas
    const deactivate = activateCanvas(canvas)

    expect(showCanvasPage('page', 4)).toBe(true)
    deactivate()

    expect(activatePage).toHaveBeenCalledWith('page', 4)
  })

  it('prepares and installs adjacent page resources before caching the frame', async () => {
    const order: string[] = []
    const prepared = {
      revision: 4,
      page: { id: 'page' },
    }
    commands.prepareCanvasPage.mockResolvedValue(prepared)
    commands.getCanvasPageManifest.mockResolvedValue(new ArrayBuffer(1))
    commands.getCanvasPageResource.mockResolvedValue(new ArrayBuffer(1))
    const canvas = {
      stageManifest: vi.fn(() => ({ token: 7, missing: ['resource'] })),
      installResource: vi.fn(async () => void order.push('resource')),
      cacheFrame: vi.fn(() => {
        order.push('frame')
        return true
      }),
    } as unknown as Canvas
    const deactivate = activateCanvas(canvas)

    const result = await prefetchCanvasPages(['page'])
    deactivate()

    expect(commands.prepareCanvasPage).toHaveBeenCalledWith('page')
    expect(commands.getCanvasPageManifest).toHaveBeenCalledWith('page', 4)
    expect(commands.getCanvasPageResource).toHaveBeenCalledWith('page', 4, 'resource')
    expect(canvas.cacheFrame).toHaveBeenCalledWith(7, 'page')
    expect(order).toEqual(['resource', 'frame'])
    expect(result).toEqual([prepared])
  })

  it('does not expose page data when the frame was rejected by the cache', async () => {
    commands.prepareCanvasPage.mockResolvedValue({ revision: 4, page: { id: 'page' } })
    commands.getCanvasPageManifest.mockResolvedValue(new ArrayBuffer(1))
    const canvas = {
      stageManifest: vi.fn(() => ({ token: 7, missing: [] })),
      cacheFrame: vi.fn(() => false),
    } as unknown as Canvas
    const deactivate = activateCanvas(canvas)

    const result = await prefetchCanvasPages(['page'])
    deactivate()

    expect(result).toEqual([])
  })

  it.each([
    ['activation', (canvas: Canvas) => canvas.activatePage('page', 4)],
    [
      'caching',
      (canvas: Canvas) => canvas.cacheFrame(canvas.stageManifest(manifest).token, 'page'),
    ],
  ])('forgets a cached page once %s reports it absent', async (_signal, report) => {
    let cached = true
    wasm.createCanvas.mockResolvedValue({
      setDeviceLostCallback: vi.fn(),
      stageManifest: vi.fn(() => ({ token: 7, missing: [] })),
      cacheFrame: vi.fn(() => cached),
      activatePage: vi.fn(() => cached),
    })
    const canvas = await createCanvas(document.createElement('canvas'), vi.fn())
    canvas.cacheFrame(canvas.stageManifest(manifest).token, 'page')

    cached = false
    report(canvas)
    // A full preparation caches the page again without passing its manifest through cacheFrame.
    cached = true
    canvas.activatePage('page', 4)

    expect(canvas.hasActiveManifest(manifest)).toBe(false)
  })
})
