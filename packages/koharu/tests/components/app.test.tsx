import { QueryClientProvider } from '@tanstack/react-query'
import { act, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { KoharuApp } from '@/components/app/KoharuApp'
import { projectKey, queryClient, useProject } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import type { ProjectInfo } from '@koharu/bridge/protocol'

const editorRender = vi.hoisted(() => vi.fn())

vi.mock('@/components/app/TitleBar', () => ({
  TitleBar: () => <output aria-label='Revision'>{useProject().data?.revision}</output>,
}))
vi.mock('@/components/editor/Editor', () => ({
  Editor: () => {
    editorRender()
    return null
  },
}))

const project: ProjectInfo = {
  name: 'Book',
  revision: 1,
  active_page: 'page',
  can_undo: false,
  can_redo: false,
}

function renderApp() {
  queryClient.setQueryData(projectKey, project)
  useKoharuStore.setState({ selectedPages: ['page'] })
  render(
    <QueryClientProvider client={queryClient}>
      <KoharuApp />
    </QueryClientProvider>,
  )
  expect(editorRender).toHaveBeenCalledTimes(1)
}

describe('KoharuApp', () => {
  it('does not re-render the editor when a commit only advances the project revision', async () => {
    renderApp()

    act(() => {
      queryClient.setQueryData(projectKey, { ...project, revision: 2, can_undo: true })
    })

    await waitFor(() =>
      expect(screen.getByRole('status', { name: 'Revision' })).toHaveTextContent('2'),
    )
    expect(editorRender).toHaveBeenCalledTimes(1)
  })

  it('does not re-render the editor when the page selection grows', () => {
    renderApp()

    act(() => {
      useKoharuStore.setState({ selectedPages: ['page', 'page-2'] })
    })

    expect(editorRender).toHaveBeenCalledTimes(1)
  })
})
