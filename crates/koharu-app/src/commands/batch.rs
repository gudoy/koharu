use std::{
    collections::BTreeMap,
    fs,
    io::Write as _,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context as _, Result, bail};
use koharu_scene::{
    AssetRole, EntityId, LanguageTag, Origin, RasterLayer, RasterLayerKind, RecognizedFrom, Region,
    RegionSpec as _, Snapshot, SourceText, TextRegion,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::{AppHandle, Manager as _, State, WebviewWindow, ipc::Channel};
use tauri_runtime_cef::CefRuntime;
use tokio::sync::watch;

use super::{
    ChannelExt as _, Error,
    lifecycle::{folder_files, import_files, replace_project},
    output::{self, ExportFormat},
    processing::{self, JobId, JobState, Processing},
    project::{CurrentProject, ProjectLibrary},
};
use koharu_desktop::Desktop;

/// Written beside the exported folders. It records each folder's outcome so a
/// rerun resumes after the last completed one and failures can be inspected.
const REPORT_FILE: &str = "koharu-batch.json";
pub(crate) const FORMATS: [ExportFormat; 2] = [ExportFormat::Psd, ExportFormat::Png];

/// Outcome of a folder in an earlier run, as recorded in the report file.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum ReportStatus {
    Running,
    Done,
    /// Exported, but some pages failed a stage; `error` lists them. A rerun
    /// skips the folder like a done one. Removing its entry retries only the
    /// unfinished pages.
    Partial,
    Failed,
    Stopped,
}

impl ReportStatus {
    /// Whether the folder was exported, even with some failed pages.
    fn exported(self) -> bool {
        matches!(self, Self::Done | Self::Partial)
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct Entry {
    status: ReportStatus,
    pages: Option<usize>,
    error: Option<String>,
    updated: u64,
}

/// Source and destination of a batch started from the command line.
pub struct Batch {
    pub source: PathBuf,
    pub destination: PathBuf,
}

/// A project folder, described before the batch starts.
#[derive(Clone, Debug, Serialize, Type)]
pub struct BatchFolder {
    pub path: String,
    pub name: String,
    #[specta(type = f64)]
    pub pages: usize,
    pub previous: Option<ReportStatus>,
    pub error: Option<String>,
    /// Whether the destination already holds this folder's export or the
    /// report records it as done.
    pub exists: bool,
}

/// What to do with a folder that was already translated into the destination.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum Existing {
    Ask,
    Replace,
    Skip,
}

#[derive(Clone, Debug, Deserialize, Type)]
pub struct BatchConfig {
    pub destination: String,
    /// Project folders to translate, in processing order. A folder's name
    /// names its project and its export.
    pub folders: Vec<String>,
    pub formats: Vec<ExportFormat>,
    /// Ends the batch at the first failed folder instead of continuing.
    pub stop_on_error: bool,
    pub existing: Existing,
}

/// What the user last asked of the running batch.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum BatchRequest {
    Run,
    Pause,
    Stop,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum BatchRun {
    Running,
    Pausing,
    Paused,
    /// Waiting for the user to decide about an already translated folder.
    Waiting,
    Stopping,
    Finished,
    Stopped,
}

impl BatchRun {
    fn active(self) -> bool {
        !matches!(self, Self::Finished | Self::Stopped)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum BatchProjectState {
    Pending,
    Processing,
    Exporting,
    Done,
    Skipped,
    Failed,
    Stopped,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct BatchProject {
    pub name: String,
    #[specta(type = f64)]
    pub pages: usize,
    pub state: BatchProjectState,
    /// Files written so far across every export format.
    #[specta(type = f64)]
    pub exported: usize,
    pub error: Option<String>,
    /// Time spent translating and exporting, pauses excluded.
    pub seconds: f64,
    /// Unix time in milliseconds when the running attempt began, so the view
    /// can add the time elapsed since the last update.
    pub since: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct BatchState {
    pub state: BatchRun,
    pub destination: String,
    #[specta(type = f64)]
    pub exports: usize,
    /// The pipeline job of the folder being processed; its progress is
    /// published on the job channel.
    pub job: Option<JobId>,
    pub projects: Vec<BatchProject>,
    /// The already translated folder awaiting a decision.
    pub conflict: Option<String>,
    pub error: Option<String>,
}

/// The user's answer for an already translated folder. `all` applies it to
/// every later folder in the same situation.
#[derive(Clone, Copy, Debug)]
struct Decision {
    replace: bool,
    all: bool,
}

pub(crate) struct BatchControl {
    pub(crate) state: Mutex<Option<BatchState>>,
    pub(crate) request: watch::Sender<BatchRequest>,
    decision: watch::Sender<Option<Decision>>,
}

impl Default for BatchControl {
    fn default() -> Self {
        Self {
            state: Mutex::new(None),
            request: watch::Sender::new(BatchRequest::Run),
            decision: watch::Sender::new(None),
        }
    }
}

#[derive(Default)]
pub(crate) struct BatchChannel {
    pub(crate) channel: Mutex<Option<Channel<BatchState>>>,
}

enum Outcome {
    /// Exported pages, and the failures of pages that could not be processed.
    Done(usize, Option<String>),
    Paused,
    Stopped,
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn pick_batch_folders(
    window: WebviewWindow<CefRuntime>,
    title: String,
    multiple: bool,
) -> std::result::Result<Vec<String>, Error> {
    let dialog = rfd::AsyncFileDialog::new()
        .set_title(title)
        .set_parent(&window);
    let folders = if multiple {
        dialog.pick_folders().await.unwrap_or_default()
    } else {
        dialog.pick_folder().await.into_iter().collect()
    };
    Ok(folders
        .iter()
        .map(|folder| folder.path().to_string_lossy().into_owned())
        .collect())
}

/// Describes project folders with their page counts and, when a destination is
/// chosen, the outcome its report recorded for each. With `expand`, every
/// subfolder of each of `folders` is a project instead of the folder itself.
#[tauri::command]
#[specta::specta]
pub(crate) async fn scan_batch(
    folders: Vec<String>,
    expand: bool,
    destination: Option<String>,
) -> std::result::Result<Vec<BatchFolder>, Error> {
    let folders = tokio::task::spawn_blocking(move || -> Result<Vec<BatchFolder>> {
        let destination = destination.map(PathBuf::from);
        let report = match &destination {
            Some(destination) => load_report(&destination.join(REPORT_FILE))?,
            None => BTreeMap::new(),
        };
        let mut projects = Vec::new();
        for folder in folders.iter().map(PathBuf::from) {
            if expand {
                projects.extend(project_folders(&folder, destination.as_deref())?);
            } else {
                projects.push((folder_name(&folder)?, folder));
            }
        }
        Ok(projects
            .into_iter()
            .map(|(name, path)| {
                let entry = report.get(&name);
                BatchFolder {
                    path: path.to_string_lossy().into_owned(),
                    exists: destination
                        .as_deref()
                        .is_some_and(|destination| translated(&report, destination, &name)),
                    pages: folder_files(&path).len(),
                    previous: entry.map(|entry| entry.status),
                    error: entry.and_then(|entry| entry.error.clone()),
                    name,
                }
            })
            .collect())
    })
    .await
    .context("folder scan failed")??;
    Ok(folders)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn start_batch(
    handle: AppHandle<CefRuntime>,
    config: BatchConfig,
) -> std::result::Result<(), Error> {
    begin(&handle, &config)?;
    drop(tauri::async_runtime::spawn(async move {
        execute(&handle, config).await;
    }));
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn control_batch(
    request: BatchRequest,
    control: State<'_, BatchControl>,
) -> std::result::Result<(), Error> {
    if !control
        .state
        .lock()
        .as_ref()
        .is_some_and(|batch| batch.state.active())
    {
        return Err(anyhow::anyhow!("no batch is running").into());
    }
    control.request.send_replace(request);
    Ok(())
}

/// Answers the pending question about an already translated folder.
#[tauri::command]
#[specta::specta]
pub(crate) async fn resolve_batch(
    replace: bool,
    all: bool,
    control: State<'_, BatchControl>,
) -> std::result::Result<(), Error> {
    if !control
        .state
        .lock()
        .as_ref()
        .is_some_and(|batch| batch.conflict.is_some())
    {
        return Err(anyhow::anyhow!("the batch is not waiting for a decision").into());
    }
    control
        .decision
        .send_replace(Some(Decision { replace, all }));
    Ok(())
}

/// Translates every folder of `source` not yet exported to `destination`, as
/// PSD and PNG. Used by the command-line batch.
pub(crate) async fn run(
    handle: &AppHandle<CefRuntime>,
    source: &Path,
    destination: &Path,
) -> Result<()> {
    let report = load_report(&destination.join(REPORT_FILE))?;
    let folders = project_folders(source, Some(destination))?
        .into_iter()
        .filter(|(name, _)| {
            report
                .get(name)
                .is_none_or(|entry| !entry.status.exported())
        })
        .map(|(_, path)| path.to_string_lossy().into_owned())
        .collect();
    let config = BatchConfig {
        destination: destination.to_string_lossy().into_owned(),
        folders,
        formats: FORMATS.to_vec(),
        stop_on_error: false,
        existing: Existing::Skip,
    };
    begin(handle, &config)?;
    execute(handle, config).await;
    Ok(())
}

/// Validates `config` and publishes the initial batch state. Holding the state
/// lock across the check keeps two batches from starting together.
fn begin(handle: &AppHandle<CefRuntime>, config: &BatchConfig) -> Result<()> {
    let destination = Path::new(&config.destination);
    if config.folders.is_empty() {
        bail!("choose at least one folder to translate");
    }
    if config.formats.is_empty() {
        bail!("choose at least one export format");
    }
    // A folder's name is its project and export name, so two folders with
    // the same name would overwrite each other.
    let mut names = std::collections::HashSet::new();
    for folder in config.folders.iter().map(Path::new) {
        if !folder.is_dir() {
            bail!("{} is not a folder", folder.display());
        }
        // Exports would otherwise land inside the folders being imported.
        if destination.starts_with(folder) {
            bail!(
                "the export destination must be outside {}",
                folder.display()
            );
        }
        let name = folder_name(folder)?;
        if !names.insert(name.clone()) {
            bail!("two folders are named {name}");
        }
    }
    if !handle.state::<Processing>().stops.lock().is_empty() {
        bail!("a batch cannot start while processing is running");
    }
    let projects = config
        .folders
        .iter()
        .map(|folder| BatchProject {
            name: folder_name(Path::new(folder)).unwrap_or_default(),
            pages: folder_files(Path::new(folder)).len(),
            state: BatchProjectState::Pending,
            exported: 0,
            error: None,
            seconds: 0.0,
            since: None,
        })
        .collect();
    let control = handle.state::<BatchControl>();
    let state = {
        let mut current = control.state.lock();
        if current.as_ref().is_some_and(|batch| batch.state.active()) {
            bail!("a batch is already running");
        }
        let state = BatchState {
            state: BatchRun::Running,
            destination: config.destination.clone(),
            exports: export_count(&config.formats),
            job: None,
            projects,
            conflict: None,
            error: None,
        };
        *current = Some(state.clone());
        state
    };
    control.request.send_replace(BatchRequest::Run);
    handle.state::<BatchChannel>().channel.publish(state);
    Ok(())
}

async fn execute(handle: &AppHandle<CefRuntime>, config: BatchConfig) {
    let result = process(handle, &config).await;
    if let Err(error) = &result {
        tracing::error!(error = format!("{error:#}"), "batch failed");
    }
    publish(handle, |batch| {
        batch.job = None;
        batch.state = if matches!(result, Ok(true)) {
            BatchRun::Finished
        } else {
            BatchRun::Stopped
        };
        batch.error = result.as_ref().err().map(|error| format!("{error:#}"));
    });
}

/// Returns whether every folder was attempted, as opposed to a stop.
async fn process(handle: &AppHandle<CefRuntime>, config: &BatchConfig) -> Result<bool> {
    let destination = Path::new(&config.destination);
    fs::create_dir_all(destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    let report_path = destination.join(REPORT_FILE);
    let mut report = load_report(&report_path)?;
    let control = handle.state::<BatchControl>();
    let mut requests = control.request.subscribe();
    let mut decisions = control.decision.subscribe();
    let mut existing = config.existing;
    tracing::info!(
        destination = %destination.display(),
        folders = config.folders.len(),
        "batch started"
    );

    for (index, folder) in config.folders.iter().map(Path::new).enumerate() {
        let name = &folder_name(folder)?;
        if translated(&report, destination, name) {
            let replace = match existing {
                Existing::Replace => true,
                Existing::Skip => false,
                Existing::Ask => {
                    let Some(decision) = ask(handle, name, &mut requests, &mut decisions).await
                    else {
                        return Ok(false);
                    };
                    if decision.all {
                        existing = if decision.replace {
                            Existing::Replace
                        } else {
                            Existing::Skip
                        };
                    }
                    decision.replace
                }
            };
            if !replace {
                tracing::info!(project = %name, "batch project skipped");
                set_project(handle, index, |project| {
                    project.state = BatchProjectState::Skipped;
                });
                continue;
            }
            remove_output(destination, name)?;
        }
        loop {
            if !wait(handle, &mut requests).await {
                return Ok(false);
            }
            set_project(handle, index, |project| {
                project.state = BatchProjectState::Processing;
                project.exported = 0;
                project.error = None;
                project.since = Some(now_millis());
            });
            update(
                &mut report,
                &report_path,
                name,
                ReportStatus::Running,
                None,
                None,
            )?;
            tracing::info!(project = %name, "batch project started");
            let started = Instant::now();
            let outcome = translate(
                handle,
                folder,
                name,
                destination,
                &config.formats,
                index,
                &mut requests,
            )
            .await;
            set_project(handle, index, |project| {
                project.seconds += started.elapsed().as_secs_f64();
                project.since = None;
            });
            match outcome {
                Ok(Outcome::Done(pages, failures)) => {
                    let status = if let Some(failures) = &failures {
                        tracing::warn!(project = %name, pages, %failures, "batch project exported with failed pages");
                        ReportStatus::Partial
                    } else {
                        tracing::info!(project = %name, pages, "batch project exported");
                        ReportStatus::Done
                    };
                    set_project(handle, index, |project| {
                        project.state = BatchProjectState::Done;
                        project.pages = pages;
                        project.error = failures.clone();
                    });
                    update(
                        &mut report,
                        &report_path,
                        name,
                        status,
                        Some(pages),
                        failures,
                    )?;
                    break;
                }
                // The folder restarts after the pause. Committed stages are
                // kept, so only the interrupted page stage runs again.
                Ok(Outcome::Paused) => {
                    set_project(handle, index, |project| {
                        project.state = BatchProjectState::Pending;
                    });
                }
                Ok(Outcome::Stopped) => {
                    tracing::info!(project = %name, "batch stopped");
                    set_project(handle, index, |project| {
                        project.state = BatchProjectState::Stopped;
                    });
                    update(
                        &mut report,
                        &report_path,
                        name,
                        ReportStatus::Stopped,
                        None,
                        None,
                    )?;
                    return Ok(false);
                }
                Err(error) => {
                    let error = format!("{error:#}");
                    tracing::error!(project = %name, %error, "batch project failed");
                    set_project(handle, index, |project| {
                        project.state = BatchProjectState::Failed;
                        project.error = Some(error.clone());
                    });
                    update(
                        &mut report,
                        &report_path,
                        name,
                        ReportStatus::Failed,
                        None,
                        Some(error),
                    )?;
                    if config.stop_on_error {
                        return Ok(false);
                    }
                    break;
                }
            }
        }
    }
    tracing::info!("batch finished");
    Ok(true)
}

/// Asks whether to replace the already translated `name`. Returns `None`
/// once the batch is stopped.
async fn ask(
    handle: &AppHandle<CefRuntime>,
    name: &str,
    requests: &mut watch::Receiver<BatchRequest>,
    decisions: &mut watch::Receiver<Option<Decision>>,
) -> Option<Decision> {
    handle.state::<BatchControl>().decision.send_replace(None);
    decisions.mark_unchanged();
    publish(handle, |batch| {
        batch.state = BatchRun::Waiting;
        batch.conflict = Some(name.to_owned());
    });
    let decision = loop {
        if *requests.borrow_and_update() == BatchRequest::Stop {
            break None;
        }
        tokio::select! {
            changed = decisions.changed() => {
                if changed.is_err() {
                    break None;
                }
                if let Some(decision) = *decisions.borrow_and_update() {
                    break Some(decision);
                }
            }
            changed = requests.changed() => {
                if changed.is_err() {
                    break None;
                }
            }
        }
    };
    publish(handle, |batch| {
        batch.conflict = None;
        batch.state = BatchRun::Running;
    });
    decision
}

/// Whether `name` already has an export in `destination`, or the report
/// records it as done.
fn translated(report: &BTreeMap<String, Entry>, destination: &Path, name: &str) -> bool {
    report
        .get(name)
        .is_some_and(|entry| entry.status.exported())
        || destination.join(format!("{name}.cbz")).is_file()
        || fs::read_dir(destination.join(name)).is_ok_and(|mut entries| entries.next().is_some())
}

/// Deletes the earlier export of `name` so a replacement holds only new files.
fn remove_output(destination: &Path, name: &str) -> Result<()> {
    let folder = destination.join(name);
    if folder.is_dir() {
        fs::remove_dir_all(&folder)
            .with_context(|| format!("failed to remove {}", folder.display()))?;
    }
    let archive = destination.join(format!("{name}.cbz"));
    if archive.is_file() {
        fs::remove_file(&archive)
            .with_context(|| format!("failed to remove {}", archive.display()))?;
    }
    Ok(())
}

/// Holds the batch while it is paused. Returns `false` once it is stopped.
async fn wait(
    handle: &AppHandle<CefRuntime>,
    requests: &mut watch::Receiver<BatchRequest>,
) -> bool {
    loop {
        let request = *requests.borrow_and_update();
        match request {
            BatchRequest::Run => {
                publish(handle, |batch| batch.state = BatchRun::Running);
                return true;
            }
            BatchRequest::Stop => return false,
            BatchRequest::Pause => {
                publish(handle, |batch| batch.state = BatchRun::Paused);
                if requests.changed().await.is_err() {
                    return false;
                }
            }
        }
    }
}

/// Runs the pipeline on `pages`, stopping the job on a pause or a stop.
/// Returns the page failures, or the outcome when the job did not finish.
async fn process_pages(
    handle: &AppHandle<CefRuntime>,
    pages: Vec<EntityId>,
    requests: &mut watch::Receiver<BatchRequest>,
) -> Result<std::result::Result<Option<String>, Outcome>> {
    let (id, mut task) = processing::start(
        handle,
        koharu_pipeline::Scope::Pages(pages),
        koharu_pipeline::Operation::Full,
        None,
    )
    .await?;
    publish(handle, |batch| batch.job = Some(id));
    let mut interrupted = false;
    let end = loop {
        tokio::select! {
            result = &mut task => break result.context("processing task failed")?,
            changed = requests.changed() => {
                let request = if changed.is_ok() { *requests.borrow() } else { BatchRequest::Stop };
                if request != BatchRequest::Run {
                    interrupted = true;
                    publish(handle, |batch| {
                        batch.state = if request == BatchRequest::Stop {
                            BatchRun::Stopping
                        } else {
                            BatchRun::Pausing
                        };
                    });
                    if let Some(stop) = handle.state::<Processing>().stops.lock().get(&id) {
                        stop.stop();
                    }
                }
            }
        }
    };
    publish(handle, |batch| batch.job = None);
    match end.state {
        JobState::Stopped => {
            if !interrupted {
                // Stopping the job itself, from the activity panel, pauses the
                // batch so it can still be resumed or stopped as a whole.
                handle
                    .state::<BatchControl>()
                    .request
                    .send_replace(BatchRequest::Pause);
            }
            Ok(Err(match *requests.borrow() {
                BatchRequest::Stop => Outcome::Stopped,
                _ => Outcome::Paused,
            }))
        }
        // Failures confined to some pages still leave a project worth exporting.
        JobState::Failed if end.failed_pages == 0 => {
            bail!(end.error.unwrap_or_else(|| "processing failed".to_owned()))
        }
        _ => Ok(Ok(end.error)),
    }
}

async fn translate(
    handle: &AppHandle<CefRuntime>,
    folder: &Path,
    name: &str,
    destination: &Path,
    formats: &[ExportFormat],
    index: usize,
    requests: &mut watch::Receiver<BatchRequest>,
) -> Result<Outcome> {
    // A rerun reopens the project so stages that already committed are skipped.
    // The project may already be the open one, which holds its write lock.
    let current = handle
        .state::<CurrentProject>()
        .project
        .lock()
        .await
        .as_ref()
        .filter(|project| project.name == name)
        .map(|project| project.snapshot().pages().next().is_none());
    let empty = match current {
        Some(empty) => empty,
        None => {
            let library = handle.state::<ProjectLibrary>().inner().clone();
            let opened = if library.exists(name)? {
                library.open(name).await?
            } else {
                library.create(name).await?
            };
            let empty = opened.snapshot().pages().next().is_none();
            replace_project(handle, opened).await?;
            empty
        }
    };
    if empty {
        import_files(handle, folder_files(folder)).await?;
    }

    let target = LanguageTag::new(
        koharu_pipeline::PipelineConfig::load()?
            .read()?
            .translation
            .target_language
            .tag(),
    )?;
    let pending = pending_pages(
        &handle
            .state::<CurrentProject>()
            .project
            .lock()
            .await
            .as_ref()
            .context("no project is open")?
            .snapshot(),
        &target,
    )?;
    let failures = if pending.is_empty() {
        None
    } else {
        match process_pages(handle, pending, requests).await? {
            Ok(failures) => failures,
            Err(outcome) => return Ok(outcome),
        }
    };

    let snapshot = handle
        .state::<CurrentProject>()
        .project
        .lock()
        .await
        .as_ref()
        .context("no project is open")?
        .snapshot();
    let pages = snapshot.pages().count();
    set_project(handle, index, |project| {
        project.state = BatchProjectState::Exporting;
        project.pages = pages;
    });
    let desktop = handle.state::<Desktop>();
    for (position, (formats, target)) in export_passes(formats, destination, name)?
        .into_iter()
        .enumerate()
    {
        let progress = |written: usize| {
            set_project(handle, index, |project| {
                project.exported = position * pages + written;
            });
        };
        let export = output::write(&desktop, snapshot.clone(), &formats, target, &progress);
        tokio::pin!(export);
        // A pause waits for the folder's export; a stop abandons it.
        loop {
            tokio::select! {
                result = &mut export => {
                    result?;
                    break;
                }
                changed = requests.changed() => {
                    let request = if changed.is_ok() { *requests.borrow() } else { BatchRequest::Stop };
                    if request == BatchRequest::Stop {
                        return Ok(Outcome::Stopped);
                    }
                    publish(handle, |batch| {
                        batch.state = if request == BatchRequest::Pause {
                            BatchRun::Pausing
                        } else {
                            BatchRun::Running
                        };
                    });
                }
            }
        }
    }
    Ok(Outcome::Done(pages, failures))
}

/// Export passes for `formats`: PNG and PSD render each page once into the
/// project folder, while a CBZ archive needs a pass of its own.
fn export_passes(
    formats: &[ExportFormat],
    destination: &Path,
    name: &str,
) -> Result<Vec<(Vec<ExportFormat>, PathBuf)>> {
    let mut passes = Vec::new();
    let files = formats
        .iter()
        .copied()
        .filter(|format| *format != ExportFormat::Cbz)
        .collect::<Vec<_>>();
    if !files.is_empty() {
        let folder = destination.join(name);
        fs::create_dir_all(&folder)
            .with_context(|| format!("failed to create {}", folder.display()))?;
        passes.push((files, folder));
    }
    if formats.contains(&ExportFormat::Cbz) {
        passes.push((
            vec![ExportFormat::Cbz],
            destination.join(format!("{name}.cbz")),
        ));
    }
    Ok(passes)
}

fn publish(handle: &AppHandle<CefRuntime>, change: impl FnOnce(&mut BatchState)) {
    let state = {
        let control = handle.state::<BatchControl>();
        let mut state = control.state.lock();
        let Some(batch) = state.as_mut() else {
            return;
        };
        change(batch);
        batch.clone()
    };
    handle.state::<BatchChannel>().channel.publish(state);
}

fn set_project(
    handle: &AppHandle<CefRuntime>,
    index: usize,
    change: impl FnOnce(&mut BatchProject),
) {
    publish(handle, |batch| {
        if let Some(project) = batch.projects.get_mut(index) {
            change(project);
        }
    });
}

/// Number of export passes `formats` take; see `export_passes`.
fn export_count(formats: &[ExportFormat]) -> usize {
    usize::from(formats.iter().any(|format| *format != ExportFormat::Cbz))
        + usize::from(formats.contains(&ExportFormat::Cbz))
}

fn now_millis() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64() * 1000.0)
}

fn folder_name(folder: &Path) -> Result<String> {
    folder
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .with_context(|| format!("{} has no valid folder name", folder.display()))
}

/// Non-hidden subfolders of `source` in natural order, excluding the
/// destination when it lives inside the source.
fn project_folders(source: &Path, destination: Option<&Path>) -> Result<Vec<(String, PathBuf)>> {
    let mut folders = fs::read_dir(source)
        .with_context(|| format!("failed to read {}", source.display()))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .map(|entry| entry.path())
        .filter(|path| Some(path.as_path()) != destination)
        .collect::<Vec<_>>();
    alphanumeric_sort::sort_path_slice(&mut folders);
    Ok(folders
        .into_iter()
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?.to_owned();
            (!name.starts_with('.')).then_some((name, path))
        })
        .collect())
}

fn load_report(path: &Path) -> Result<BTreeMap<String, Entry>> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse {}", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(error) => Err(error).with_context(|| format!("failed to read {}", path.display())),
    }
}

fn update(
    report: &mut BTreeMap<String, Entry>,
    path: &Path,
    name: &str,
    status: ReportStatus,
    pages: Option<usize>,
    error: Option<String>,
) -> Result<()> {
    let updated = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    report.insert(
        name.to_owned(),
        Entry {
            status,
            pages,
            error,
            updated,
        },
    );
    let directory = path.parent().context("report path has no parent")?;
    let mut file = tempfile::NamedTempFile::new_in(directory)?;
    file.write_all(&serde_json::to_vec_pretty(report)?)?;
    file.persist(path)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// Pages a batch run still sends through the pipeline. OCR and translation
/// have no skip of their own, so a resumed run would otherwise redo every page
/// that finished before the interruption.
fn pending_pages(snapshot: &Snapshot, target: &LanguageTag) -> Result<Vec<EntityId>> {
    let mut pending = Vec::new();
    for page in snapshot.pages() {
        if !finished(snapshot, page.id(), target)? {
            pending.push(page.id());
        }
    }
    Ok(pending)
}

/// A page is finished only when every stage left its output, read the way the
/// stage itself reads the scene: a detected text region, source text on every
/// content recognized from one, a translation in the target language for every
/// non-blank source, and a cleanup layer with an image. User-authored text
/// counts because no stage rewrites it. A generated translation of a
/// user-edited source does not: the scene does not record whether the edit came
/// after it. A page without detected text cannot be told apart from an
/// unprocessed one, so it is not finished.
fn finished(snapshot: &Snapshot, page: EntityId, target: &LanguageTag) -> Result<bool> {
    let mut detected = false;
    for entity in snapshot.descendants(page)? {
        if !entity
            .component::<Region>()?
            .is_some_and(|region| region.kind == TextRegion::kind())
        {
            continue;
        }
        detected = true;
        for relation in snapshot.relations_to_as::<RecognizedFrom>(entity.id()) {
            if snapshot
                .component::<SourceText>(relation.value().source)?
                .is_none()
            {
                return Ok(false);
            }
        }
    }
    if !detected {
        return Ok(false);
    }
    if let Some(group) = snapshot.page(page)?.text_group()? {
        for layer in group.text_layers()? {
            let content = layer.content()?;
            let Some(source) = content.source()? else {
                continue;
            };
            if source.text.value.trim().is_empty() {
                continue;
            }
            let translated = content.translation()?.is_some_and(|translation| {
                matches!(translation.text.origin, Origin::User)
                    || (!matches!(source.text.origin, Origin::User)
                        && translation.language.as_ref() == Some(target))
            });
            if !translated {
                return Ok(false);
            }
        }
    }
    let image = AssetRole::new("source")?;
    for child in snapshot.children(page)? {
        if snapshot
            .component::<RasterLayer>(child)?
            .is_some_and(|layer| layer.kind == RasterLayerKind::Cleanup)
            && snapshot.asset(child, &image)?.is_some()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use koharu_scene::{
        AssetInput, AssetMetadata, AssetRole, At, Authored, Edit, Generation, Geometry, Origin,
        PageDraft, ProducerId, RasterLayer, RasterLayerKind, RecognizedFrom, Session, SourceText,
        TextLayout, TextLayoutKind, TextRegion, Translation,
    };

    use super::*;

    /// Adds a page and, when `cleaned`, the cleanup layer inpainting leaves.
    fn add_page(edit: &mut Edit, cleaned: bool) -> EntityId {
        let page = edit
            .add_page(PageDraft::new("page", 10.0, 10.0), At::End)
            .unwrap();
        if cleaned {
            let cleanup = edit.add_entity(page, At::Start).unwrap();
            edit.set(
                cleanup,
                &RasterLayer {
                    origin: Origin::User,
                    name: "Cleanup".to_owned(),
                    kind: RasterLayerKind::Cleanup,
                },
            )
            .unwrap();
            edit.set_asset(
                cleanup,
                &AssetRole::new("source").unwrap(),
                AssetInput::new(
                    b"clean".as_slice(),
                    "image/png",
                    AssetMetadata {
                        width: Some(1),
                        height: Some(1),
                        attributes: Default::default(),
                    },
                ),
            )
            .unwrap();
        }
        page
    }

    /// Adds a detected text region to `page` and returns its text content.
    fn add_text(edit: &mut Edit, page: EntityId) -> EntityId {
        let region = edit
            .add_analysis_region::<TextRegion>(
                page,
                At::End,
                &Geometry::rectangle(1.0, 1.0, 4.0, 4.0),
                None,
            )
            .unwrap();
        let content = edit.add_text_content(page, At::End).unwrap();
        edit.add_text_layer(
            page,
            At::End,
            content,
            &TextLayout {
                origin: Origin::User,
                kind: TextLayoutKind::Paragraph,
                angle_degrees: None,
            },
        )
        .unwrap();
        edit.relate::<RecognizedFrom>(content, region).unwrap();
        content
    }

    // The edit stamps the origin: a plain edit is the user's, `edit_as` a stage's.
    fn source(text: &str) -> SourceText {
        SourceText {
            text: Authored::user(text.to_owned()),
            language: None,
        }
    }

    fn translation(text: &str, language: Option<&str>) -> Translation {
        Translation {
            text: Authored::user(text.to_owned()),
            language: language.map(|tag| LanguageTag::new(tag).unwrap()),
        }
    }

    #[tokio::test]
    async fn resume_reprocesses_only_pages_a_stage_has_not_finished() {
        let mut session = Session::memory().await.unwrap();
        let mut edit = session.snapshot().edit();
        let done = add_page(&mut edit, true);
        let untranslated = add_page(&mut edit, true);
        let other_language = add_page(&mut edit, true);
        let unread = add_page(&mut edit, true);
        let blank = add_page(&mut edit, true);
        let edited = add_page(&mut edit, true);
        let stale = add_page(&mut edit, true);
        let not_cleaned = add_page(&mut edit, false);
        let no_text = add_page(&mut edit, true);
        let done_text = add_text(&mut edit, done);
        let untranslated_text = add_text(&mut edit, untranslated);
        let other_language_text = add_text(&mut edit, other_language);
        add_text(&mut edit, unread);
        let blank_text = add_text(&mut edit, blank);
        let edited_text = add_text(&mut edit, edited);
        let stale_text = add_text(&mut edit, stale);
        let not_cleaned_text = add_text(&mut edit, not_cleaned);
        // Neither OCR nor translation rewrites user-authored text.
        edit.set(edited_text, &source("こんにちは")).unwrap();
        edit.set(edited_text, &translation("Hi", None)).unwrap();
        // A generated translation may predate the user's edit of its source.
        edit.set(stale_text, &source("こんばんは")).unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        let mut edit = session.snapshot().edit_as(Generation::new(
            ProducerId::new("dev.koharu.pipeline.test").unwrap(),
        ));
        for content in [
            done_text,
            untranslated_text,
            other_language_text,
            not_cleaned_text,
        ] {
            edit.set(content, &source("こんにちは")).unwrap();
        }
        // Translation skips blank sources, so none is expected.
        edit.set(blank_text, &source("  ")).unwrap();
        for content in [done_text, stale_text, not_cleaned_text] {
            edit.set(content, &translation("Hello", Some("en")))
                .unwrap();
        }
        edit.set(other_language_text, &translation("Bonjour", Some("fr")))
            .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        assert_eq!(
            pending_pages(&session.snapshot(), &LanguageTag::new("en").unwrap()).unwrap(),
            [
                untranslated,
                other_language,
                unread,
                stale,
                not_cleaned,
                no_text
            ]
        );
    }
}
