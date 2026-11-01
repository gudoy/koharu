'use client'

import { revealItemsInDir } from '@tauri-apps/plugin-opener'
import type { TFunction } from 'i18next'
import {
  CircleAlert,
  CircleCheck,
  CircleDashed,
  CircleMinus,
  CircleSlash,
  Clock,
  FolderOpen,
  FolderPlus,
  FolderTree,
  LoaderCircle,
  Pause,
  Play,
  Square,
  X,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { call } from '@/lib/backend'
import { useKoharuStore } from '@/lib/store'
import {
  commands,
  type BatchFolder,
  type BatchProject,
  type BatchState,
  type Existing,
  type ExportFormat,
  type Job,
} from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import { Checkbox } from '@koharu/ui/components/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@koharu/ui/components/dialog'
import { RadioGroup, RadioGroupItem } from '@koharu/ui/components/radio-group'
import { cn } from '@koharu/ui/lib/utils'

const formats: ExportFormat[] = ['psd', 'png', 'cbz']

export function BatchDialog() {
  const { t } = useTranslation()
  const open = useKoharuStore((state) => state.batchOpen)
  const setOpen = useKoharuStore((state) => state.setBatchOpen)
  const batch = useKoharuStore((state) => state.batch)

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className='flex max-h-[min(44rem,calc(100dvh-2rem))] flex-col gap-4 p-5 sm:max-w-3xl'>
        <DialogHeader className='gap-1'>
          <DialogTitle className='text-[15px]'>{t('batch.title')}</DialogTitle>
          <DialogDescription className='text-[11px]'>{t('batch.description')}</DialogDescription>
        </DialogHeader>
        {batch ? <BatchProgress batch={batch} /> : <BatchSetup />}
      </DialogContent>
    </Dialog>
  )
}

function BatchSetup() {
  const { t } = useTranslation()
  const setOpen = useKoharuStore((state) => state.setBatchOpen)
  const [destination, setDestination] = useState<string | null>(null)
  const [folders, setFolders] = useState<BatchFolder[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [scanning, setScanning] = useState(false)
  const [chosenFormats, setChosenFormats] = useState<Set<ExportFormat>>(new Set(['psd', 'png']))
  const [stopOnError, setStopOnError] = useState(false)
  const [existing, setExisting] = useState<Existing>('ask')
  const [starting, setStarting] = useState(false)

  const scan = async (paths: string[], expand: boolean, target: string | null) => {
    setScanning(true)
    try {
      return await call(commands.scanBatch, paths, expand, target)
    } finally {
      setScanning(false)
    }
  }

  // Each folder's export status depends on the destination, so the list is
  // described again whenever it changes.
  const paths = folders.map((folder) => folder.path)
  useEffect(() => {
    if (paths.length === 0) return
    let active = true
    void scan(paths, false, destination)
      .then((found) => {
        if (active) setFolders(found)
      })
      .catch(() => undefined)
    return () => {
      active = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [destination])

  // `expand` adds the subfolders of each chosen folder, each as a project.
  const add = async (expand: boolean) => {
    const picked = await call(
      commands.pickBatchFolders,
      t(expand ? 'batch.addParentDialog' : 'batch.addProjectsDialog'),
      true,
    ).catch(() => [])
    if (picked.length === 0) return
    const found = await scan(picked, expand, destination).catch(() => [])
    const known = new Set(paths)
    const added = found.filter((folder) => !known.has(folder.path))
    setFolders((current) => [...current, ...added])
    // Already translated folders stay selected: the existing-project choice
    // decides whether they are skipped or replaced.
    setSelected((current) => new Set([...current, ...added.map((folder) => folder.path)]))
  }

  const remove = (path: string) => {
    setFolders((current) => current.filter((folder) => folder.path !== path))
    setSelected((current) => toggle(current, path, false))
  }

  const chooseDestination = async () => {
    const [path] = await call(commands.pickBatchFolders, t('batch.destinationDialog'), false).catch(
      () => [],
    )
    if (path) setDestination(path)
  }

  const chosen = folders.filter((folder) => selected.has(folder.path))
  const pages = chosen.reduce((sum, folder) => sum + folder.pages, 0)
  const translated = chosen.filter((folder) => folder.exists).length
  // Folders skipped as already translated cost no time.
  const pagesToTranslate = chosen
    .filter((folder) => existing !== 'skip' || !folder.exists)
    .reduce((sum, folder) => sum + folder.pages, 0)
  const rate = useMemo(readRate, [])
  const nameCounts = new Map<string, number>()
  for (const folder of chosen) nameCounts.set(folder.name, (nameCounts.get(folder.name) ?? 0) + 1)
  const duplicated = (folder: BatchFolder) =>
    selected.has(folder.path) && (nameCounts.get(folder.name) ?? 0) > 1
  const hasDuplicates = chosen.some(duplicated)
  // Exports inside a project folder would be imported by a later run.
  const insideProject =
    destination !== null &&
    chosen.some(
      (folder) => destination === folder.path || destination.startsWith(`${folder.path}/`),
    )
  const ready =
    destination !== null &&
    !insideProject &&
    !hasDuplicates &&
    chosen.length > 0 &&
    chosenFormats.size > 0 &&
    !scanning

  const start = () => {
    if (!ready) return
    setStarting(true)
    void call(commands.startBatch, {
      destination,
      folders: chosen.map((folder) => folder.path),
      formats: formats.filter((format) => chosenFormats.has(format)),
      stop_on_error: stopOnError,
      existing,
    })
      .catch(() => undefined)
      .finally(() => setStarting(false))
  }

  return (
    <>
      <div className='grid min-h-0 flex-1 gap-4 sm:grid-cols-[minmax(0,17rem)_minmax(0,1fr)]'>
        <div className='grid content-start gap-4'>
          <Section title={t('batch.source')}>
            <div className='grid gap-2'>
              <Button
                size='sm'
                variant='outline'
                className='justify-start'
                disabled={scanning}
                onClick={() => void add(false)}
              >
                <FolderPlus aria-hidden='true' />
                {t('batch.addProjects')}
              </Button>
              <Button
                size='sm'
                variant='outline'
                className='justify-start'
                disabled={scanning}
                onClick={() => void add(true)}
              >
                <FolderTree aria-hidden='true' />
                {t('batch.addParent')}
              </Button>
            </div>
            <p className='text-[10px] text-muted-foreground'>{t('batch.sourceHint')}</p>
          </Section>
          <Section title={t('batch.destination')}>
            <PathField
              path={destination}
              placeholder={t('batch.noFolder')}
              action={t('batch.choose')}
              onChoose={() => void chooseDestination()}
            />
            <p
              className={cn(
                'text-[10px]',
                insideProject ? 'text-destructive' : 'text-muted-foreground',
              )}
            >
              {insideProject ? t('batch.sameFolder') : t('batch.destinationHint')}
            </p>
          </Section>
          <Section title={t('batch.formats')}>
            <div className='flex flex-wrap gap-x-4 gap-y-2'>
              {formats.map((format) => (
                <label key={format} className='flex items-center gap-2 text-[11px]'>
                  <Checkbox
                    checked={chosenFormats.has(format)}
                    onCheckedChange={(checked) =>
                      setChosenFormats((current) => toggle(current, format, checked))
                    }
                  />
                  {format.toUpperCase()}
                </label>
              ))}
            </div>
          </Section>
          <Section title={t('batch.existing')}>
            <RadioGroup
              value={existing}
              onValueChange={(value) => setExisting(value as Existing)}
              className='gap-2'
            >
              {(['ask', 'skip', 'replace'] as const).map((policy) => (
                <label key={policy} className='flex items-center gap-2 text-[11px]'>
                  <RadioGroupItem value={policy} />
                  {t(`batch.existingPolicy.${policy}`)}
                </label>
              ))}
            </RadioGroup>
            {translated > 0 && (
              <p className='text-[10px] text-muted-foreground'>
                {t('batch.existingCount', { count: translated })}
              </p>
            )}
          </Section>
          <Section title={t('batch.errors')}>
            <RadioGroup
              value={stopOnError ? 'stop' : 'continue'}
              onValueChange={(value) => setStopOnError(value === 'stop')}
              className='gap-2'
            >
              <label className='flex items-center gap-2 text-[11px]'>
                <RadioGroupItem value='continue' />
                {t('batch.continueOnError')}
              </label>
              <label className='flex items-center gap-2 text-[11px]'>
                <RadioGroupItem value='stop' />
                {t('batch.stopOnError')}
              </label>
            </RadioGroup>
          </Section>
        </div>

        <div className='flex min-h-0 flex-col rounded-xl border border-border/80 bg-[var(--surface-panel)]'>
          <div className='flex items-center gap-2 border-b px-3 py-2'>
            <span className='text-[10px] font-semibold tracking-[0.1em] uppercase'>
              {t('batch.projects')}
            </span>
            {folders.length > 0 && (
              <span className='text-[10px] text-muted-foreground tabular-nums'>
                {t('batch.selection', { count: chosen.length, total: folders.length, pages })}
              </span>
            )}
            <div className='flex-1' />
            {folders.length > 0 && (
              <>
                <Button size='xs' variant='ghost' onClick={() => setSelected(new Set(paths))}>
                  {t('batch.selectAll')}
                </Button>
                <Button size='xs' variant='ghost' onClick={() => setSelected(new Set())}>
                  {t('batch.selectNone')}
                </Button>
                <Button
                  size='xs'
                  variant='ghost'
                  onClick={() => {
                    setFolders([])
                    setSelected(new Set())
                  }}
                >
                  {t('batch.clear')}
                </Button>
              </>
            )}
          </div>
          <div className='min-h-40 flex-1 overflow-y-auto'>
            {scanning ? (
              <Placeholder>
                <LoaderCircle className='size-3.5 animate-spin' aria-hidden='true' />
                {t('batch.scanning')}
              </Placeholder>
            ) : folders.length === 0 ? (
              <Placeholder>{t('batch.chooseSource')}</Placeholder>
            ) : (
              <ul className='divide-y'>
                {folders.map((folder) => (
                  <li key={folder.path} className='flex items-center pr-1.5'>
                    <label className='flex min-w-0 flex-1 items-center gap-2.5 py-2 pl-3 text-[11px]'>
                      <Checkbox
                        checked={selected.has(folder.path)}
                        onCheckedChange={(checked) =>
                          setSelected((current) => toggle(current, folder.path, checked))
                        }
                      />
                      <span className='min-w-0 flex-1 truncate' title={folder.path}>
                        {folder.name}
                      </span>
                      {duplicated(folder) ? (
                        <span className='shrink-0 rounded-full bg-destructive/10 px-1.5 py-0.5 text-[9px] text-destructive'>
                          {t('batch.duplicate')}
                        </span>
                      ) : folder.exists ? (
                        <span className='shrink-0 rounded-full bg-primary/10 px-1.5 py-0.5 text-[9px] text-primary'>
                          {t('batch.previous.done')}
                        </span>
                      ) : (
                        folder.previous &&
                        folder.previous !== 'running' &&
                        folder.previous !== 'done' && (
                          <span
                            className={cn(
                              'shrink-0 rounded-full px-1.5 py-0.5 text-[9px]',
                              folder.previous === 'failed'
                                ? 'bg-destructive/10 text-destructive'
                                : 'bg-muted text-muted-foreground',
                            )}
                            title={folder.error ?? undefined}
                          >
                            {t(`batch.previous.${folder.previous}`)}
                          </span>
                        )
                      )}
                      <span className='w-16 shrink-0 text-right text-muted-foreground tabular-nums'>
                        {t('batch.pages', { count: folder.pages })}
                      </span>
                    </label>
                    <Button
                      size='icon-xs'
                      variant='ghost'
                      aria-label={t('batch.remove', { name: folder.name })}
                      onClick={() => remove(folder.path)}
                    >
                      <X />
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>

      <DialogFooter className='items-center'>
        {pagesToTranslate > 0 && (
          <p className='mr-auto flex items-center gap-1.5 text-[11px] text-muted-foreground'>
            <Clock className='size-3.5' aria-hidden='true' />
            {rate === null
              ? t('batch.estimateUnknown')
              : t('batch.estimate', {
                  duration: formatDuration(t, pagesToTranslate * rate),
                  rate: formatDuration(t, rate),
                })}
          </p>
        )}
        <Button variant='outline' size='sm' onClick={() => setOpen(false)}>
          {t('batch.cancel')}
        </Button>
        <Button size='sm' disabled={!ready || starting} onClick={start}>
          {starting ? (
            <LoaderCircle className='animate-spin' aria-hidden='true' />
          ) : (
            <Play aria-hidden='true' />
          )}
          {t('batch.start', { count: chosen.length, pages })}
        </Button>
      </DialogFooter>
    </>
  )
}

function BatchProgress({ batch }: { batch: BatchState }) {
  const { t, i18n } = useTranslation()
  const job = useKoharuStore((state) => (batch.job ? state.jobs[batch.job] : undefined))
  const dismiss = useKoharuStore((state) => state.dismissBatch)
  const setOpen = useKoharuStore((state) => state.setBatchOpen)
  const summary = useMemo(() => summarize(batch, job), [batch, job])
  const active = !['finished', 'stopped'].includes(batch.state)
  const now = useNow(active)
  const timing = timeBatch(batch, job, summary.pagesDone, now)
  // Each finished project refines the rate the next batches start from.
  useEffect(() => {
    if (timing.measured !== null) writeRate(timing.measured)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [summary.done])
  const paused = batch.state === 'paused' || batch.state === 'pausing'
  const control = (request: 'run' | 'pause' | 'stop') =>
    void call(commands.controlBatch, request).catch(() => undefined)

  return (
    <>
      <div className='grid gap-2 rounded-xl border border-border/80 bg-[var(--surface-panel)] p-3'>
        <div className='flex flex-wrap items-baseline gap-x-3 gap-y-1'>
          <span
            className={cn(
              'rounded-full px-2 py-0.5 text-[10px] font-medium',
              batch.state === 'running'
                ? 'bg-primary/10 text-primary'
                : batch.state === 'finished'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground',
            )}
          >
            {t(`batch.state.${batch.state}`)}
          </span>
          <span className='text-[12px] font-medium tabular-nums'>
            {t('batch.projectsProgress', { done: summary.done, total: summary.projects })}
          </span>
          <span className='text-[11px] text-muted-foreground tabular-nums'>
            {t('batch.pagesProgress', {
              done: Math.round(summary.pagesDone),
              total: summary.pages,
            })}
          </span>
          {summary.skipped > 0 && (
            <span className='text-[11px] text-muted-foreground tabular-nums'>
              {t('batch.skippedCount', { count: summary.skipped })}
            </span>
          )}
          {summary.failed > 0 && (
            <span className='text-[11px] text-destructive tabular-nums'>
              {t('batch.failedCount', { count: summary.failed })}
            </span>
          )}
          <span className='ml-auto text-[12px] font-semibold tabular-nums'>{summary.percent}%</span>
        </div>
        <Bar value={summary.percent} className='h-2' />
        <div className='flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground tabular-nums'>
          <span className='flex items-center gap-1.5'>
            <Clock className='size-3.5' aria-hidden='true' />
            {t('batch.elapsed', { duration: formatDuration(t, timing.elapsed) })}
          </span>
          {active &&
            (timing.remaining === null ? (
              <span>{t('batch.remainingUnknown')}</span>
            ) : (
              <>
                <span className='font-medium text-foreground'>
                  {t('batch.remaining', { duration: formatDuration(t, timing.remaining) })}
                </span>
                {!paused && batch.state !== 'waiting' && (
                  <span>
                    {t('batch.finishAt', {
                      time: new Date(now + timing.remaining * 1000).toLocaleTimeString(
                        i18n.language,
                        { hour: '2-digit', minute: '2-digit' },
                      ),
                    })}
                  </span>
                )}
              </>
            ))}
        </div>
        <p className='truncate text-[10px] text-muted-foreground' title={batch.destination}>
          {t('batch.savedTo', { path: batch.destination })}
        </p>
        {batch.error && <p className='text-[11px] text-destructive'>{batch.error}</p>}
      </div>

      {batch.conflict && <ConflictPrompt name={batch.conflict} />}

      <ul className='min-h-0 flex-1 divide-y overflow-y-auto rounded-xl border border-border/80'>
        {batch.projects.map((project) => (
          <ProjectRow
            key={project.name}
            project={project}
            job={project.state === 'processing' ? job : undefined}
            exports={batch.exports}
            destination={batch.destination}
          />
        ))}
      </ul>

      <DialogFooter>
        {active ? (
          <>
            <Button
              variant='outline'
              size='sm'
              disabled={batch.state === 'stopping'}
              onClick={() => control('stop')}
            >
              <Square className='fill-current' aria-hidden='true' />
              {t('batch.stopAll')}
            </Button>
            {paused ? (
              <Button
                size='sm'
                disabled={batch.state === 'stopping'}
                onClick={() => control('run')}
              >
                <Play aria-hidden='true' />
                {t('batch.resume')}
              </Button>
            ) : (
              <Button
                size='sm'
                disabled={batch.state === 'stopping' || batch.state === 'waiting'}
                onClick={() => control('pause')}
              >
                <Pause aria-hidden='true' />
                {t('batch.pause')}
              </Button>
            )}
          </>
        ) : (
          <>
            <Button
              variant='outline'
              size='sm'
              onClick={() => void revealItemsInDir(batch.destination).catch(() => undefined)}
            >
              <FolderOpen aria-hidden='true' />
              {t('batch.openDestination')}
            </Button>
            <Button variant='outline' size='sm' onClick={dismiss}>
              {t('batch.newBatch')}
            </Button>
            <Button
              size='sm'
              onClick={() => {
                setOpen(false)
                dismiss()
              }}
            >
              {t('batch.close')}
            </Button>
          </>
        )}
      </DialogFooter>
    </>
  )
}

function ConflictPrompt({ name }: { name: string }) {
  const { t } = useTranslation()
  const [all, setAll] = useState(false)
  const [sent, setSent] = useState(false)
  const resolve = (replace: boolean) => {
    setSent(true)
    void call(commands.resolveBatch, replace, all)
      .catch(() => undefined)
      .finally(() => setSent(false))
  }
  return (
    <div
      role='alertdialog'
      aria-labelledby='batch-conflict'
      className='grid gap-3 rounded-xl border border-primary/40 bg-primary/5 p-3'
    >
      <p id='batch-conflict' className='text-[12px]'>
        {t('batch.conflict')} <span className='font-medium break-all'>{name}</span>
      </p>
      <div className='flex flex-wrap items-center gap-2'>
        <label className='mr-auto flex items-center gap-2 text-[11px]'>
          <Checkbox checked={all} onCheckedChange={setAll} />
          {t('batch.applyToAll')}
        </label>
        <Button size='sm' variant='outline' disabled={sent} onClick={() => resolve(false)}>
          {t('batch.skip')}
        </Button>
        <Button size='sm' disabled={sent} onClick={() => resolve(true)}>
          {t('batch.replace')}
        </Button>
      </div>
    </div>
  )
}

function ProjectRow({
  project,
  job,
  exports,
  destination,
}: {
  project: BatchProject
  job: Job | undefined
  exports: number
  destination: string
}) {
  const { t } = useTranslation()
  const progress = Math.round(projectFraction(project, job, exports) * 100)
  const detail =
    project.state === 'processing'
      ? job?.stage
        ? t('batch.stage', {
            stage: t(`phase.${job.stage}`, { defaultValue: job.stage }),
            done: job.completed,
            total: job.total,
          })
        : t('batch.preparing')
      : project.state === 'exporting'
        ? t('batch.exporting', { done: project.exported, total: project.pages * exports })
        : null

  return (
    <li className='grid grid-cols-[1rem_minmax(0,1fr)_auto] items-center gap-x-2.5 px-3 py-2 text-[11px]'>
      <StateIcon state={project.state} />
      <span className='truncate font-medium' title={project.name}>
        {project.name}
      </span>
      <span className='flex items-center gap-2 text-muted-foreground tabular-nums'>
        {t('batch.pages', { count: project.pages })}
        <span
          className={cn(
            'w-20 text-right',
            project.state === 'failed' && 'text-destructive',
            project.state === 'done' && 'text-primary',
          )}
        >
          {t(`batch.projectState.${project.state}`)}
        </span>
        {project.state === 'done' && (
          <Button
            size='icon-xs'
            variant='ghost'
            aria-label={t('batch.openProject')}
            onClick={() =>
              void revealItemsInDir(`${destination}/${project.name}`).catch(() => undefined)
            }
          >
            <FolderOpen />
          </Button>
        )}
      </span>
      {detail && (
        <div className='col-start-2 col-end-4 grid gap-1 pt-1'>
          <span className='truncate text-[10px] text-muted-foreground tabular-nums'>{detail}</span>
          <Bar value={progress} />
        </div>
      )}
      {project.error && (
        <p className='col-start-2 col-end-4 max-h-16 overflow-y-auto pt-1 text-[10px] break-words text-destructive select-text'>
          {project.error}
        </p>
      )}
    </li>
  )
}

function StateIcon({ state }: { state: BatchProject['state'] }) {
  const className = 'size-3.5 justify-self-center'
  switch (state) {
    case 'processing':
    case 'exporting':
      return <LoaderCircle className={cn(className, 'animate-spin text-primary')} />
    case 'done':
      return <CircleCheck className={cn(className, 'text-primary')} />
    case 'failed':
      return <CircleAlert className={cn(className, 'text-destructive')} />
    case 'skipped':
      return <CircleMinus className={cn(className, 'text-muted-foreground')} />
    case 'stopped':
      return <CircleSlash className={cn(className, 'text-muted-foreground')} />
    default:
      return <CircleDashed className={cn(className, 'text-muted-foreground/60')} />
  }
}

function toggle<T>(set: Set<T>, value: T, checked: boolean): Set<T> {
  const next = new Set(set)
  if (checked) next.add(value)
  else next.delete(value)
  return next
}
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className='grid gap-2'>
      <h3 className='text-[10px] font-semibold tracking-[0.1em] text-muted-foreground uppercase'>
        {title}
      </h3>
      {children}
    </section>
  )
}

function PathField({
  path,
  placeholder,
  action,
  onChoose,
}: {
  path: string | null
  placeholder: string
  action: string
  onChoose: () => void
}) {
  return (
    <div className='flex items-center gap-2'>
      {/* Truncated from the start so the folder name stays visible. */}
      <span
        className={cn(
          'min-w-0 flex-1 truncate rounded-md border bg-background px-2 py-1.5 text-left text-[11px] [direction:rtl]',
          !path && 'text-muted-foreground',
        )}
        title={path ?? undefined}
      >
        <bdi>{path ?? placeholder}</bdi>
      </span>
      <Button size='sm' variant='outline' onClick={onChoose}>
        {action}
      </Button>
    </div>
  )
}

function Placeholder({ children }: { children: React.ReactNode }) {
  return (
    <div className='flex h-full min-h-40 items-center justify-center gap-2 p-4 text-center text-[11px] text-muted-foreground'>
      {children}
    </div>
  )
}

function Bar({ value, className }: { value: number; className?: string }) {
  return (
    <div
      role='progressbar'
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={value}
      className={cn('h-1 overflow-hidden rounded-full bg-muted', className)}
    >
      <div
        className='h-full rounded-full bg-primary transition-[width] duration-300'
        style={{ width: `${value}%` }}
      />
    </div>
  )
}

// Translation dominates a folder's run, so it counts for most of the folder's
// progress and the export for the rest.
const PROCESSING_SHARE = 0.9

function projectFraction(project: BatchProject, job: Job | undefined, exports: number): number {
  switch (project.state) {
    case 'done':
      return 1
    case 'processing':
      return job && job.total > 0 ? (job.completed / job.total) * PROCESSING_SHARE : 0
    case 'exporting': {
      // Each export pass writes every page once.
      const total = project.pages * exports
      return (
        PROCESSING_SHARE + (total > 0 ? (project.exported / total) * (1 - PROCESSING_SHARE) : 0)
      )
    }
    default:
      return 0
  }
}

function summarize(batch: BatchState, job: Job | undefined) {
  // Skipped folders were translated before, so they leave the totals.
  const projects = batch.projects.filter((project) => project.state !== 'skipped')
  const pages = projects.reduce((sum, project) => sum + project.pages, 0)
  const pagesDone = projects.reduce(
    (sum, project) =>
      sum +
      project.pages *
        projectFraction(project, project.state === 'processing' ? job : undefined, batch.exports),
    0,
  )
  return {
    pages,
    pagesDone,
    projects: projects.length,
    done: projects.filter((project) => project.state === 'done').length,
    skipped: batch.projects.length - projects.length,
    failed: projects.filter((project) => project.state === 'failed').length,
    percent: pages > 0 ? Math.min(100, Math.round((pagesDone / pages) * 100)) : 0,
  }
}

// Seconds per page measured by the last batches. The speed depends on the
// machine and the chosen models, so it stays on this device.
const RATE_KEY = 'koharu.batch.secondsPerPage'
// Below this, setup and model loading dominate and the rate is noise.
const MIN_MEASURE_SECONDS = 30
const MIN_MEASURE_PAGES = 1

function readRate(): number | null {
  try {
    const rate = Number(localStorage.getItem(RATE_KEY))
    return rate > 0 ? rate : null
  } catch {
    return null
  }
}

function writeRate(rate: number) {
  try {
    localStorage.setItem(RATE_KEY, String(rate))
  } catch {
    // The estimate is a convenience: without storage, the next batch measures again.
  }
}

/** Current time, refreshed every second while `active`. */
function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [active])
  return now
}

function timeBatch(batch: BatchState, job: Job | undefined, pagesDone: number, now: number) {
  const elapsed = batch.projects.reduce(
    (sum, project) =>
      sum +
      project.seconds +
      (project.since === null ? 0 : Math.max(0, now - project.since) / 1000),
    0,
  )
  const measured =
    elapsed >= MIN_MEASURE_SECONDS && pagesDone >= MIN_MEASURE_PAGES ? elapsed / pagesDone : null
  const rate = measured ?? readRate()
  // Failed and stopped projects will not run again in this batch.
  const remainingPages = batch.projects.reduce((sum, project) => {
    if (!['pending', 'processing', 'exporting'].includes(project.state)) return sum
    const fraction = projectFraction(
      project,
      project.state === 'processing' ? job : undefined,
      batch.exports,
    )
    return sum + project.pages * (1 - fraction)
  }, 0)
  return { elapsed, measured, remaining: rate === null ? null : remainingPages * rate }
}

function formatDuration(t: TFunction, seconds: number): string {
  const minutes = Math.round(seconds / 60)
  if (minutes < 1) return t('batch.duration.seconds', { count: Math.max(1, Math.round(seconds)) })
  if (minutes < 60) return t('batch.duration.minutes', { count: minutes })
  const hours = Math.floor(minutes / 60)
  return t('batch.duration.hours', { hours, minutes: String(minutes % 60).padStart(2, '0') })
}
