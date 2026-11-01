'use client'

import { useEffect, useRef, useState, type ComponentProps } from 'react'

import { Textarea } from '@koharu/ui/components/textarea'

type CommitTextareaProps = Omit<ComponentProps<typeof Textarea>, 'value' | 'onChange'> & {
  value: string
  delay?: number
  /**
   * Called with `draft` set after a typing pause, for a live preview. One edit
   * session ends with a single call without it: on blur, Mod+Enter, keypad
   * Enter, or unmount.
   */
  onCommit: (value: string, draft: boolean) => void
}

export function CommitTextarea({ value, delay = 360, onCommit, ...props }: CommitTextareaProps) {
  const [draft, setDraft] = useState(value)
  const timer = useRef<number | null>(null)
  const composing = useRef(false)
  const external = useRef(value)
  const latest = useRef(value)
  const previewed = useRef(false)
  const committer = useRef(onCommit)

  useEffect(() => {
    committer.current = onCommit
  })

  useEffect(() => {
    external.current = value
    if (!composing.current && timer.current === null) {
      latest.current = value
      setDraft(value)
    }
  }, [value])

  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current)
      if (previewed.current || latest.current !== external.current) {
        committer.current(latest.current, false)
      }
    },
    [],
  )

  const preview = (next: string) => {
    timer.current = null
    if (next === external.current) return
    previewed.current = true
    committer.current(next, true)
  }

  const commit = (next: string) => {
    if (timer.current !== null) window.clearTimeout(timer.current)
    timer.current = null
    if (!previewed.current && next === external.current) return
    previewed.current = false
    committer.current(next, false)
  }

  const schedule = (next: string) => {
    if (timer.current !== null) window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => preview(next), delay)
  }

  const edit = (next: string) => {
    latest.current = next
    setDraft(next)
  }

  return (
    <Textarea
      {...props}
      value={draft}
      onChange={(event) => {
        const next = event.currentTarget.value
        edit(next)
        if (!composing.current) schedule(next)
      }}
      onCompositionStart={() => {
        composing.current = true
      }}
      onCompositionEnd={(event) => {
        composing.current = false
        const next = event.currentTarget.value
        edit(next)
        schedule(next)
      }}
      onKeyDown={(event) => {
        if (event.key !== 'Enter' || event.nativeEvent.isComposing) return
        if (!event.metaKey && !event.ctrlKey && event.code !== 'NumpadEnter') return
        event.preventDefault()
        commit(event.currentTarget.value)
      }}
      onBlur={() => commit(draft)}
    />
  )
}
