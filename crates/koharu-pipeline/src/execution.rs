use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};

use anyhow::{Context as _, Result, ensure};
use koharu_scene::{EntityId, Snapshot};
use tokio::task::JoinSet;
use tracing::Instrument as _;

use crate::{
    Committer, ErrorKind, PipelineError, Progress, ProgressSink, Report, Request, RunStatus, Stage,
    StageOutput, StopToken,
    images::ImageCache,
    progress,
    resources::ResourceMonitor,
    scheduler::Scheduler,
    scope::NormalizedScope,
    stage_runner::{StageCompletion, StageJob, StageOutcome, StageRunner},
    stages::StageInput,
};

pub(crate) struct Execution<'a> {
    runner: Arc<StageRunner>,
    resources: Arc<ResourceMonitor>,
    committer: &'a mut dyn Committer,
    stop: StopToken,
    progress: Option<ProgressSink>,
    scope: NormalizedScope,
    scheduler: Scheduler,
    scene: Snapshot,
    images: BTreeMap<EntityId, Arc<ImageCache>>,
    busy_stages: BTreeSet<Stage>,
    completed: usize,
    failure: Option<PipelineError>,
    base: koharu_scene::Revision,
    started: Instant,
    inpainting_mask: Option<crate::InpaintingMask>,
}

impl<'a> Execution<'a> {
    pub(crate) fn new(
        runner: Arc<StageRunner>,
        resources: Arc<ResourceMonitor>,
        snapshot: Snapshot,
        request: Request,
        committer: &'a mut dyn Committer,
    ) -> std::result::Result<Self, PipelineError> {
        let started = Instant::now();
        let base = snapshot.revision();
        let stages = request
            .operation
            .stages()
            .map_err(|error| PipelineError::new(ErrorKind::InvalidInput, None, error))?;
        let scope = NormalizedScope::new(&snapshot, &request.scope, &stages)
            .map_err(|error| PipelineError::new(ErrorKind::InvalidInput, None, error))?;
        let pages = scope.pages().to_vec();
        if let Some(mask) = request.inpainting_mask.as_ref()
            && (!pages.contains(&mask.page) || !stages.contains(&Stage::Inpainting))
        {
            return Err(PipelineError::new(
                ErrorKind::InvalidInput,
                Some(Stage::Inpainting),
                anyhow::anyhow!("the inpainting mask page is outside the inpainting scope"),
            ));
        }
        progress::emit(
            request.progress.as_ref(),
            Progress::Started {
                pages: pages.clone(),
                stages: stages.clone(),
            },
        );

        Ok(Self {
            runner,
            resources,
            committer,
            stop: request.stop,
            progress: request.progress,
            scope,
            scheduler: Scheduler::new(&pages, &stages),
            scene: snapshot,
            images: BTreeMap::new(),
            busy_stages: BTreeSet::new(),
            completed: 0,
            failure: None,
            base,
            started,
            inpainting_mask: request.inpainting_mask,
        })
    }

    pub(crate) async fn run(mut self) -> std::result::Result<Report, PipelineError> {
        if self.stopped() {
            return Ok(self.report(RunStatus::Stopped));
        }

        self.resources.start();
        self.resources.wait_for_sample().await;

        // Each stage runs as its own task so it keeps progressing while this loop awaits a
        // commit. Commits stay serialized here. Dropping the set aborts the stages in flight.
        let mut running = JoinSet::new();
        loop {
            while let Some(job) = self.take_ready_job() {
                let runner = self.runner.clone();
                running.spawn(async move { runner.run(job).await }.in_current_span());
            }

            let Some(joined) = running.join_next().await else {
                break;
            };
            let completion =
                joined.unwrap_or_else(|error| std::panic::resume_unwind(error.into_panic()));
            self.busy_stages.remove(&completion.stage);
            if self.stopped() || self.failure.is_some() {
                continue;
            }
            if let Err(error) = self.apply_completion(completion).await {
                self.failure = Some(error);
            }
        }

        self.finalize()
    }

    fn take_ready_job(&mut self) -> Option<StageJob> {
        if self.stopped() || self.failure.is_some() {
            return None;
        }
        let (page, stage) = self.scheduler.start_next(&self.busy_stages)?;
        self.busy_stages.insert(stage);
        let images = self
            .images
            .entry(page)
            .or_insert_with(|| Arc::new(ImageCache::default()))
            .clone();
        Some(StageJob::new(
            stage,
            StageInput::new(
                self.scene.clone(),
                page,
                self.scope.entities(),
                self.scope.region(page),
                images,
                self.inpainting_mask
                    .as_ref()
                    .filter(|mask| stage == Stage::Inpainting && mask.page == page)
                    .cloned(),
                self.stop.clone(),
            ),
            self.progress.clone(),
        ))
    }

    async fn apply_completion(
        &mut self,
        completion: StageCompletion,
    ) -> std::result::Result<(), PipelineError> {
        let StageCompletion {
            page,
            stage,
            model,
            elapsed,
            outcome,
        } = completion;
        match outcome? {
            StageOutcome::Stopped => {}
            StageOutcome::Skipped => {
                self.mark_complete(page, stage);
                progress::emit(self.progress.as_ref(), Progress::Skipped { page, stage });
            }
            StageOutcome::Patch(patch) => {
                if !self.commit_patch(page, stage, patch).await? {
                    return Ok(());
                }
                self.mark_complete(page, stage);
                progress::emit(
                    self.progress.as_ref(),
                    Progress::Finished {
                        page,
                        stage,
                        model,
                        elapsed,
                    },
                );
            }
        }
        Ok(())
    }

    async fn commit_patch(
        &mut self,
        page: EntityId,
        stage: Stage,
        patch: koharu_scene::Patch,
    ) -> std::result::Result<bool, PipelineError> {
        let patch = patch
            .rebase_on(&self.scene)
            .and_then(|patch| {
                patch.validate_on(&self.scene)?;
                Ok(patch.with_label(format!("Pipeline {stage} for page {page}")))
            })
            .context("failed to rebase stage output onto the latest scene")
            .map_err(|error| PipelineError::new(ErrorKind::InvalidOutput, Some(stage), error))?;
        if self.stopped() {
            return Ok(false);
        }

        let next = self
            .committer
            .commit(StageOutput { page, stage, patch })
            .await
            .with_context(|| format!("failed to commit {stage} output for page {page}"))
            .map_err(|error| PipelineError::new(ErrorKind::Commit, Some(stage), error))?;
        validate_commit(&self.scene, &next)
            .map_err(|error| PipelineError::new(ErrorKind::Commit, Some(stage), error))?;
        self.scene = next;
        Ok(true)
    }

    fn mark_complete(&mut self, page: EntityId, stage: Stage) {
        if self.scheduler.complete_stage(page, stage) {
            self.images.remove(&page);
        }
        self.completed += 1;
    }

    fn stopped(&self) -> bool {
        self.stop.stopped()
    }

    fn finalize(mut self) -> std::result::Result<Report, PipelineError> {
        if let Some(error) = self.failure.take() {
            return Err(error);
        }
        if !self.stopped() && self.completed != self.scheduler.total() {
            return Err(PipelineError::new(
                ErrorKind::InvalidOutput,
                None,
                anyhow::anyhow!(
                    "pipeline scheduler stopped after {} of {} work items",
                    self.completed,
                    self.scheduler.total()
                ),
            ));
        }
        let status = if self.stopped() {
            RunStatus::Stopped
        } else {
            RunStatus::Completed
        };
        Ok(self.report(status))
    }

    fn report(&self, status: RunStatus) -> Report {
        Report {
            status,
            base: self.base,
            final_revision: self.scene.revision(),
            completed: self.completed,
            total: self.scheduler.total(),
            elapsed: self.started.elapsed(),
        }
    }
}

fn validate_commit(previous: &Snapshot, next: &Snapshot) -> Result<()> {
    ensure!(
        previous.project_id() == next.project_id(),
        "committer returned a snapshot from another project"
    );
    ensure!(
        next.revision() > previous.revision(),
        "committer did not advance the scene revision"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    use koharu_scene::{At, PageDraft, Patch, Session};

    use super::*;
    use crate::{Operation, PipelineConfig, stages::StageProcessor};

    /// Adds one entity to its page. Each call takes longer than the one before, so two stages in
    /// flight never finish at the same instant. The third call cannot finish before a commit is
    /// pending, so that commit always overlaps a stage in flight regardless of timer jitter.
    struct Delayed {
        started: AtomicUsize,
        finished: tokio::sync::watch::Sender<usize>,
        commit_pending: tokio::sync::Notify,
    }

    #[async_trait::async_trait]
    impl StageProcessor for Delayed {
        fn model(&self) -> &'static str {
            "delayed"
        }

        fn unload(&self) -> bool {
            false
        }

        async fn load(&self) -> Result<()> {
            Ok(())
        }

        async fn process(&self, input: StageInput) -> Result<Patch> {
            let call = self.started.fetch_add(1, Ordering::SeqCst) + 1;
            if call == 3 {
                self.commit_pending.notified().await;
            }
            tokio::time::sleep(Duration::from_millis(50 * call as u64)).await;
            let patch = input.scene().patch(|edit| {
                edit.add_entity(input.page(), At::End)?;
                Ok(())
            })?;
            self.finished.send_modify(|finished| *finished += 1);
            Ok(patch)
        }
    }

    /// Before each commit, waits for the stages still in flight to finish their processing.
    struct WaitingCommitter {
        session: Session,
        processor: Arc<Delayed>,
        waits: Vec<bool>,
    }

    #[async_trait::async_trait]
    impl Committer for WaitingCommitter {
        async fn commit(&mut self, output: StageOutput) -> Result<Snapshot> {
            let started = self.processor.started.load(Ordering::SeqCst);
            let mut finished = self.processor.finished.subscribe();
            if *finished.borrow() < started {
                self.processor.commit_pending.notify_one();
                let progressed = tokio::time::timeout(
                    Duration::from_secs(2),
                    finished.wait_for(|finished| *finished == started),
                )
                .await
                .is_ok();
                self.waits.push(progressed);
            }
            Ok(self.session.commit(output.patch).await?.snapshot)
        }
    }

    #[tokio::test]
    async fn stages_in_flight_keep_running_while_a_commit_is_pending() {
        let device = koharu_ml::Device::cpu();
        let translator = koharu_translator::Translator::from_config(
            device.clone(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        let resources = ResourceMonitor::new(&device);
        let mut runner = StageRunner::new(
            &PipelineConfig::default(),
            translator,
            &device,
            resources.clone(),
        )
        .unwrap();
        let processor = Arc::new(Delayed {
            started: AtomicUsize::new(0),
            finished: tokio::sync::watch::channel(0).0,
            commit_pending: tokio::sync::Notify::new(),
        });
        runner.set_test_processor(processor.clone());
        let mut session = Session::memory().await.unwrap();
        let pages = session
            .snapshot()
            .patch(|edit| {
                edit.add_page(PageDraft::new("one", 1.0, 1.0), At::End)?;
                edit.add_page(PageDraft::new("two", 1.0, 1.0), At::End)?;
                Ok(())
            })
            .unwrap();
        session.commit(pages).await.unwrap();
        let snapshot = session.snapshot();
        let mut committer = WaitingCommitter {
            session,
            processor,
            waits: Vec::new(),
        };
        let request = Request {
            operation: Operation::Through {
                stage: Stage::Inpainting,
            },
            ..Request::default()
        };

        let report = Execution::new(
            Arc::new(runner),
            resources,
            snapshot,
            request,
            &mut committer,
        )
        .unwrap()
        .run()
        .await
        .unwrap();

        assert_eq!(report.completed, 4);
        assert!(
            !committer.waits.is_empty(),
            "no commit overlapped a stage in flight"
        );
        assert!(
            committer.waits.iter().all(|progressed| *progressed),
            "a stage in flight stalled while a commit was pending"
        );
    }
}
