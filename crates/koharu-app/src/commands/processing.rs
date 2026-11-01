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

use super::{
    ChannelExt as _, Error, canvas::CanvasChannel, lifecycle::Initialization,
    project::CurrentProject,
};
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
    Waiting,
    Running,
    Finished,
    Failed,
    Stopped,
}

#[derive(Default)]
pub(crate) struct Processing {
    pub(crate) stops: Mutex<HashMap<JobId, StopToken>>,
    pub(crate) jobs: Mutex<HashMap<JobId, Job>>,
}

impl Processing {
    /// Stops every job and returns its terminal update. A job task that ends
    /// later finds no entry and publishes nothing, so the caller must publish
    /// these updates.
    pub(crate) fn stop_all(&self) -> Vec<Job> {
        for (_, stop) in self.stops.lock().drain() {
            stop.stop();
        }
        self.jobs
            .lock()
            .drain()
            .map(|(_, mut job)| {
                job.state = JobState::Stopped;
                job
            })
            .collect()
    }
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
    let (id, _) = start(&handle, scope, operation, None).await?;
    Ok(id)
}

/// Starts a pipeline job. The returned task resolves to the job's terminal
/// state once the run has published its final update. The inpainting mask
/// belongs to this job only, so a rejected start cannot leave it for a later
/// run.
pub(crate) async fn start(
    handle: &AppHandle<CefRuntime>,
    scope: koharu_pipeline::Scope,
    operation: koharu_pipeline::Operation,
    inpainting_mask: Option<koharu_pipeline::InpaintingMask>,
) -> Result<(JobId, JoinHandle<(JobState, Option<String>)>)> {
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
    let waiting = !handle.state::<Initialization>().is_ready();
    let job = Job {
        id,
        state: if waiting {
            JobState::Waiting
        } else {
            JobState::Running
        },
        completed: 0,
        total: 0,
        page: None,
        stage: None,
        model: None,
        error: None,
    };
    processing.jobs.lock().insert(id, job.clone());
    job_channel.channel.publish(job);

    let task_handle = handle.clone();
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
            let update = match event {
                Progress::Started { pages, stages } => {
                    let mut progress = progress.lock();
                    *progress = (0, pages.len().saturating_mul(stages.len()));
                    Some((0, progress.1, None, None, None))
                }
                Progress::Loading { page, stage, model } => {
                    let progress = progress.lock();
                    Some((progress.0, progress.1, Some(page), Some(stage), Some(model)))
                }
                Progress::Finished {
                    page, stage, model, ..
                } => {
                    let mut progress = progress.lock();
                    progress.0 = progress.0.saturating_add(1).min(progress.1);
                    Some((progress.0, progress.1, Some(page), Some(stage), Some(model)))
                }
                Progress::Skipped { page, stage } => {
                    let mut progress = progress.lock();
                    progress.0 = progress.0.saturating_add(1).min(progress.1);
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
        // A job started before the ML runtime is ready waits for it, then
        // processes the project as it is at that point.
        let ready = async {
            if !waiting {
                return Ok(Some(snapshot));
            }
            if !task_handle
                .state::<Initialization>()
                .wait_unless(&stop)
                .await?
            {
                return Ok(None);
            }
            let snapshot = task_handle
                .state::<CurrentProject>()
                .project
                .lock()
                .await
                .as_ref()
                .context("no project is open")?
                .snapshot();
            // Closing or switching the project stops jobs before replacing it.
            if stop.stopped() {
                return Ok(None);
            }
            let job = task_handle
                .state::<Processing>()
                .jobs
                .lock()
                .get_mut(&id)
                .map(|job| {
                    job.state = JobState::Running;
                    job.clone()
                });
            if let Some(job) = job {
                task_handle.state::<JobChannel>().channel.publish(job);
            }
            Ok::<_, anyhow::Error>(Some(snapshot))
        };
        let (stopped, error) = match ready.await {
            Ok(None) => (true, None),
            Err(error) => (false, Some(format!("{error:#}"))),
            Ok(Some(snapshot)) => {
                let pipeline = task_handle.state::<koharu_pipeline::Pipeline>();
                match pipeline.execute(snapshot, request, &mut committer).await {
                    Ok(report) => (report.status == RunStatus::Stopped, None),
                    Err(error) => {
                        tracing::error!(stage = ?error.stage, %error, "processing failed");
                        (false, Some(format!("{error:#}")))
                    }
                }
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
        (state, error)
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

#[cfg(test)]
mod tests {
    use koharu_pipeline::StopToken;

    use super::{Job, JobId, JobState, Processing};

    #[test]
    fn stopping_all_jobs_reports_a_waiting_job_as_stopped() {
        let processing = Processing::default();
        let id = JobId::new();
        let stop = StopToken::default();
        processing.stops.lock().insert(id, stop.clone());
        processing.jobs.lock().insert(
            id,
            Job {
                id,
                state: JobState::Waiting,
                completed: 0,
                total: 0,
                page: None,
                stage: None,
                model: None,
                error: None,
            },
        );

        let stopped = processing.stop_all();

        assert!(stop.stopped());
        assert!(processing.stops.lock().is_empty());
        assert!(processing.jobs.lock().is_empty());
        assert_eq!(
            stopped
                .iter()
                .map(|job| (job.id, job.state))
                .collect::<Vec<_>>(),
            [(id, JobState::Stopped)]
        );
    }
}
