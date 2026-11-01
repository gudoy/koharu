use std::{collections::HashMap, fmt, sync::Arc};

use anyhow::{Context as _, Result};
use koharu_pipeline::{Committer, Progress, RunStatus, StageOutput, StopToken};
use koharu_scene::Snapshot;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use specta::Type;
use tauri::{AppHandle, Manager as _, State, ipc::Channel};
use tauri_runtime_cef::CefRuntime;
use tokio::task::JoinHandle;
use uuid::Uuid;

use super::{ChannelExt as _, Error, canvas::CanvasChannel, project::CurrentProject};
use koharu_desktop::Desktop;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize, Type)]
#[serde(transparent)]
pub struct JobId(Uuid);

impl JobId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct Job {
    pub id: JobId,
    pub state: JobState,
    #[specta(type = f64)]
    pub completed: usize,
    #[specta(type = f64)]
    pub total: usize,
    pub page: Option<koharu_scene::EntityId>,
    pub stage: Option<koharu_pipeline::Stage>,
    pub model: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Running,
    Finished,
    Failed,
    Stopped,
}

/// How a pipeline job ended. A job whose failures stayed confined to their
/// pages is `Failed` with `failed_pages > 0`; every other page was processed.
pub(crate) struct JobEnd {
    pub(crate) state: JobState,
    pub(crate) error: Option<String>,
    pub(crate) failed_pages: usize,
}

#[derive(Default)]
pub(crate) struct Processing {
    pub(crate) stops: Mutex<HashMap<JobId, StopToken>>,
    pub(crate) jobs: Mutex<HashMap<JobId, Job>>,
    pub(crate) inpainting_mask: Mutex<Option<koharu_pipeline::InpaintingMask>>,
}

#[derive(Default)]
pub(crate) struct JobChannel {
    pub(crate) channel: Mutex<Option<Channel<Job>>>,
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn process(
    handle: AppHandle<CefRuntime>,
    scope: koharu_pipeline::Scope,
    operation: koharu_pipeline::Operation,
) -> std::result::Result<JobId, Error> {
    let (id, _) = start(&handle, scope, operation).await?;
    Ok(id)
}

/// Starts a pipeline job. The returned task resolves to the job's terminal
/// state once the run has published its final update.
pub(crate) async fn start(
    handle: &AppHandle<CefRuntime>,
    scope: koharu_pipeline::Scope,
    operation: koharu_pipeline::Operation,
) -> Result<(JobId, JoinHandle<JobEnd>)> {
    let project = handle.state::<CurrentProject>();
    let processing = handle.state::<Processing>();
    let job_channel = handle.state::<JobChannel>();
    let snapshot = project
        .project
        .lock()
        .await
        .as_ref()
        .context("no project is open")?
        .snapshot();
    let id = JobId::new();
    let stop = StopToken::default();
    {
        let mut stops = processing.stops.lock();
        if !stops.is_empty() {
            anyhow::bail!("another process is already running");
        }
        stops.insert(id, stop.clone());
    }
    let job = Job {
        id,
        state: JobState::Running,
        completed: 0,
        total: 0,
        page: None,
        stage: None,
        model: None,
        error: None,
    };
    processing.jobs.lock().insert(id, job.clone());
    job_channel.channel.publish(job);

    let pipeline = handle.state::<koharu_pipeline::Pipeline>().inner().clone();
    let task_handle = handle.clone();
    let inpainting_mask = processing.inpainting_mask.lock().take();
    let task = tokio::spawn(async move {
        let progress = Arc::new(Mutex::new((0_usize, 0_usize)));
        let progress_handle = task_handle.clone();
        let mut request = koharu_pipeline::Request {
            operation,
            scope,
            stop: stop.clone(),
            progress: None,
            inpainting_mask,
        };
        request.progress = Some(Arc::new(move |event| {
            // Stage tasks emit concurrently with the run loop. Hold the counter until the job is
            // written and published so an older count never replaces a newer one.
            let mut progress = progress.lock();
            let update = match event {
                Progress::Started { pages, stages } => {
                    *progress = (0, pages.len().saturating_mul(stages.len()));
                    Some((0, progress.1, None, None, None))
                }
                Progress::Loading { page, stage, model } => {
                    Some((progress.0, progress.1, Some(page), Some(stage), Some(model)))
                }
                Progress::Finished {
                    page, stage, model, ..
                } => {
                    progress.0 = progress.0.saturating_add(1).min(progress.1);
                    Some((progress.0, progress.1, Some(page), Some(stage), Some(model)))
                }
                Progress::Skipped { page, stage } => {
                    progress.0 = progress.0.saturating_add(1).min(progress.1);
                    Some((progress.0, progress.1, Some(page), Some(stage), None))
                }
                Progress::Failed {
                    page,
                    stage,
                    abandoned,
                } => {
                    progress.0 = progress
                        .0
                        .saturating_add(1 + abandoned.len())
                        .min(progress.1);
                    Some((progress.0, progress.1, Some(page), Some(stage), None))
                }
                Progress::Running { .. } => None,
            };
            if let Some((completed, total, page, stage, model)) = update {
                let job = {
                    let processing = progress_handle.state::<Processing>();
                    let mut jobs = processing.jobs.lock();
                    jobs.get_mut(&id).map(|job| {
                        job.completed = completed;
                        job.total = total;
                        job.page = page;
                        job.stage = stage;
                        job.model = model;
                        job.clone()
                    })
                };
                if let Some(job) = job {
                    progress_handle.state::<JobChannel>().channel.publish(job);
                }
            }
        }));

        struct PipelineCommitter {
            handle: AppHandle<CefRuntime>,
        }

        #[async_trait::async_trait]
        impl Committer for PipelineCommitter {
            async fn commit(&mut self, output: StageOutput) -> Result<Snapshot> {
                let (commit, page) = {
                    let projects = self.handle.state::<CurrentProject>();
                    let mut projects = projects.project.lock().await;
                    let project = projects.as_mut().context("no project is open")?;
                    let Some(commit) = project.commit_rebased(output.patch).await? else {
                        return Ok(project.snapshot());
                    };
                    project.record_commit(&commit);
                    let page = project.active_page();
                    (commit, page)
                };
                let snapshot = commit.snapshot.clone();
                let desktop = self.handle.state::<Desktop>();
                desktop.synchronize(&commit.snapshot, page, &commit).await?;
                let canvas = desktop.canvas_state();
                self.handle.state::<CanvasChannel>().channel.publish(canvas);
                Ok(snapshot)
            }
        }

        let mut committer = PipelineCommitter {
            handle: task_handle.clone(),
        };
        let pages = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
        let result = pipeline.execute(snapshot, request, &mut committer).await;
        let (stopped, error, failed_pages) = match result {
            Ok(report) => {
                let error = (!report.failures.is_empty()).then(|| {
                    report
                        .failures
                        .iter()
                        .map(|failure| {
                            let number = pages
                                .iter()
                                .position(|page| *page == failure.page)
                                .map_or_else(
                                    || failure.page.to_string(),
                                    |index| (index + 1).to_string(),
                                );
                            format!("page {number}: {:#}", failure.error)
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                });
                if let Some(error) = &error {
                    tracing::error!(pages = report.failures.len(), %error, "processing failed on some pages");
                }
                (
                    report.status == RunStatus::Stopped,
                    error,
                    report.failures.len(),
                )
            }
            Err(error) => {
                tracing::error!(stage = ?error.stage, %error, "processing failed");
                (false, Some(format!("{error:#}")), 0)
            }
        };
        task_handle.state::<Processing>().stops.lock().remove(&id);
        let state = if stopped {
            JobState::Stopped
        } else if error.is_some() {
            JobState::Failed
        } else {
            JobState::Finished
        };
        let job = task_handle
            .state::<Processing>()
            .jobs
            .lock()
            .remove(&id)
            .map(|mut job| {
                job.state = state;
                job.error.clone_from(&error);
                job
            });
        if let Some(job) = job {
            task_handle.state::<JobChannel>().channel.publish(job);
        }
        JobEnd {
            state,
            error,
            failed_pages,
        }
    });
    Ok((id, task))
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn stop_job(
    job: JobId,
    processing: State<'_, Processing>,
) -> std::result::Result<(), Error> {
    let stops = processing.stops.lock();
    let stop = stops
        .get(&job)
        .with_context(|| format!("job {job} is not running"))?;
    stop.stop();
    Ok(())
}
