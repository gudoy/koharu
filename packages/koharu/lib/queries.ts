'use client'

import {
  QueryClient,
  type QueryKey,
  queryOptions,
  useIsMutating,
  useMutation,
  useQuery,
} from '@tanstack/react-query'

import { commands, type FontFamily, type Job, type Page, type ProjectInfo } from '@koharu/bridge/protocol'

import { call } from './backend'

export const projectKey = ['project'] as const
export const pagesKey = ['pages'] as const
export const pageKey = ['page'] as const
export const preparedPageKey = (page: string) => ['prepared-page', page] as const
export const fontsKey = ['fonts'] as const

const projectQuery = queryOptions({
  queryKey: projectKey,
  queryFn: () => call(commands.getProject),
})

const pagesQuery = queryOptions({
  queryKey: pagesKey,
  queryFn: () => call(commands.getPages),
})

const pageQuery = queryOptions({
  queryKey: pageKey,
  queryFn: () => call(commands.getPage),
})

const fontsQuery = queryOptions({
  queryKey: fontsKey,
  queryFn: () => call(commands.getFonts),
})

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: Number.POSITIVE_INFINITY,
      retry: false,
      refetchOnReconnect: false,
      refetchOnWindowFocus: false,
    },
  },
})

export function useProject(enabled = true) {
  return useQuery({ ...projectQuery, enabled })
}

/** Subscribes to a projection of the project, so a commit that leaves it equal does not re-render. */
export function useProjectSelect<T>(select: (project: ProjectInfo | null) => T) {
  return useQuery({ ...projectQuery, select }).data
}

export function usePages(enabled = true) {
  return useQuery({ ...pagesQuery, enabled })
}

export function usePage(enabled = true) {
  return useQuery({ ...pageQuery, enabled })
}

/** Subscribes to a projection of the open page, so an edit that leaves it equal does not re-render. */
export function usePageSelect<T>(select: (page: Page | null) => T) {
  return useQuery({ ...pageQuery, select }).data
}

export function useFonts(enabled = true) {
  return useQuery({ ...fontsQuery, enabled })
}

export function useFontPreview(font: FontFamily | undefined, enabled = true) {
  return useQuery({
    queryKey: ['font-preview', font?.name],
    queryFn: async () => {
      if (!font) return null
      try {
        return new Uint8Array(await commands.getFontPreview(font.name))
      } catch {
        return null
      }
    },
    enabled: enabled && font !== undefined,
    gcTime: 5 * 60 * 1000,
  })
}

export function useCommand<Args extends unknown[], Result>(
  key: QueryKey,
  command: (...args: Args) => Promise<Result>,
  label: string,
  onSuccess?: () => Promise<void>,
) {
  const busy = useIsMutating({ mutationKey: key }) > 0
  const mutation = useMutation({
    mutationKey: key,
    mutationFn: (args: Args) => call(command, ...args),
    meta: { activity: label },
    onSuccess,
  })
  return { run: (...args: Args) => mutation.mutate(args), busy }
}

export function useImportPages() {
  const { run, busy } = useCommand(['import-pages'], commands.import, 'navigator.importing', () =>
    refresh(projectKey, pagesKey, pageKey),
  )
  return { importPages: run, importing: busy }
}

export async function refresh(...keys: QueryKey[]): Promise<void> {
  await Promise.all(keys.map((queryKey) => queryClient.invalidateQueries({ queryKey })))
}

/**
 * Refreshes project queries as a job reports progress. A long job reports a
 * step per stage per page, so the page list is refreshed at most once per
 * `interval` while it runs, and the open page only when a step touched it.
 */
export function createJobRefresh(interval = 1000) {
  const completed = new Map<string, number>()
  let pagesTimer: ReturnType<typeof setTimeout> | null = null
  let pagesPending = false

  const refreshPages = () => {
    if (pagesTimer) {
      pagesPending = true
      return
    }
    void refresh(pagesKey).catch(() => undefined)
    pagesTimer = setTimeout(() => {
      pagesTimer = null
      if (pagesPending) {
        pagesPending = false
        refreshPages()
      }
    }, interval)
  }

  return (job: Job) => {
    const previous = completed.get(job.id) ?? 0
    if (job.state !== 'running') {
      completed.delete(job.id)
      if (pagesTimer) clearTimeout(pagesTimer)
      pagesTimer = null
      pagesPending = false
      void refresh(projectKey, pagesKey, pageKey).catch(() => undefined)
      return
    }
    completed.set(job.id, job.completed)
    if (job.completed <= previous) return
    void refresh(projectKey).catch(() => undefined)
    refreshPages()
    const open = queryClient.getQueryData<Page | null>(pageKey)?.id
    if (job.page === null || job.page === open) {
      void refresh(pageKey).catch(() => undefined)
    }
  }
}
