import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createJobRefresh, pageKey, pagesKey, projectKey, queryClient } from '@/lib/queries'
import type { Job } from '@koharu/bridge/protocol'

function job(completed: number, page: string, state: Job['state'] = 'running'): Job {
  return { id: 'job', state, completed, total: 100, page, stage: 'ocr', model: null, error: null }
}

function invalidated(invalidate: ReturnType<typeof vi.spyOn>, key: readonly string[]) {
  return invalidate.mock.calls.filter(([filters]) => filters?.queryKey === key).length
}

describe('job progress refresh', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    queryClient.setQueryData(pageKey, { id: 'active' })
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('refreshes the open page only for its own steps and throttles the page list', () => {
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries').mockResolvedValue()
    const receive = createJobRefresh(1000)

    for (let completed = 1; completed <= 20; completed++) receive(job(completed, 'other'))

    expect(invalidated(invalidate, pageKey)).toBe(0)
    expect(invalidated(invalidate, projectKey)).toBe(20)
    expect(invalidated(invalidate, pagesKey)).toBe(1)
    vi.advanceTimersByTime(1000)
    expect(invalidated(invalidate, pagesKey)).toBe(2)

    receive(job(21, 'active'))
    expect(invalidated(invalidate, pageKey)).toBe(1)
  })

  it('refreshes everything when the job ends', () => {
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries').mockResolvedValue()
    const receive = createJobRefresh(1000)

    receive(job(1, 'other'))
    receive(job(2, 'other'))
    receive(job(2, 'other', 'finished'))

    expect(invalidated(invalidate, pageKey)).toBe(1)
    expect(invalidated(invalidate, pagesKey)).toBe(2)
    // The throttled page-list refresh is already covered by the final one.
    vi.advanceTimersByTime(1000)
    expect(invalidated(invalidate, pagesKey)).toBe(2)
  })

  it('does not treat a job waiting for the runtime as ended', () => {
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries').mockResolvedValue()
    const receive = createJobRefresh(1000)

    receive(job(0, 'active', 'waiting'))

    expect(invalidate).not.toHaveBeenCalled()
  })

  it('ignores progress events that complete nothing', () => {
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries').mockResolvedValue()
    const receive = createJobRefresh(1000)

    receive(job(0, 'active'))

    expect(invalidate).not.toHaveBeenCalled()
  })
})
