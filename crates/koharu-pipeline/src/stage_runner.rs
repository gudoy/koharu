use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Result;
use koharu_scene::{EntityId, Patch};

use crate::{
    ErrorKind, PipelineConfig, PipelineError, Progress, ProgressSink, Stage,
    accelerator::AcceleratorGate,
    progress,
    resources::ResourceMonitor,
    stages::{StageInput, Stages},
};

pub(crate) struct StageRunner {
    stages: Stages,
    accelerator: AcceleratorGate,
}

impl StageRunner {
    pub(crate) fn new(
        config: &PipelineConfig,
        translator: koharu_translator::Translator,
        device: &koharu_ml::Device,
        resources: Arc<ResourceMonitor>,
    ) -> Result<Self> {
        Ok(Self {
            stages: Stages::new(config, translator, device)?,
            accelerator: AcceleratorGate::new(device, resources),
        })
    }

    /// Builds the runner for a changed configuration. Unchanged stages keep
    /// their processors and loaded models, and the runner keeps the same
    /// accelerator lane.
    pub(crate) fn reconfigure(
        &self,
        config: &PipelineConfig,
        translator: koharu_translator::Translator,
        device: &koharu_ml::Device,
    ) -> Result<Self> {
        Ok(Self {
            stages: self.stages.reconfigure(config, translator, device)?,
            accelerator: self.accelerator.clone(),
        })
    }

    #[cfg(test)]
    pub(crate) fn shares_processor(&self, other: &Self, stage: Stage) -> bool {
        self.stages.shares_processor(&other.stages, stage)
    }

    #[cfg(test)]
    pub(crate) fn set_test_processor(&mut self, processor: Arc<dyn crate::stages::StageProcessor>) {
        self.stages.set_test_processor(processor);
    }

    #[tracing::instrument(skip_all)]
    pub(crate) async fn run(&self, job: StageJob) -> StageCompletion {
        let started = Instant::now();
        let page = job.input.page();
        let model = self.stages.model(job.stage).to_owned();
        let outcome = self.run_with_recovery(&job, &model).await;
        StageCompletion {
            page,
            stage: job.stage,
            model,
            elapsed: started.elapsed(),
            outcome,
        }
    }

    async fn run_with_recovery(
        &self,
        job: &StageJob,
        model: &str,
    ) -> std::result::Result<StageOutcome, PipelineError> {
        if job.input.stopped() {
            return Ok(StageOutcome::Stopped);
        }
        let skip = self.stages.skip(job.stage, &job.input).map_err(|error| {
            self.stage_error(
                job.stage,
                model,
                AttemptFailure {
                    kind: ErrorKind::Processing,
                    error,
                },
            )
        })?;
        if skip {
            return Ok(StageOutcome::Skipped);
        }
        let permit = self.accelerator.acquire().await;
        if job.input.stopped() {
            return Ok(StageOutcome::Stopped);
        }
        let first = self.load_and_process(job, model).await;
        let failure = match first {
            Ok(outcome) => return Ok(outcome),
            Err(failure) if is_out_of_memory(&failure.error) && !job.input.stopped() => failure,
            Err(failure) => return Err(self.stage_error(job.stage, model, failure)),
        };

        drop(permit);
        tracing::warn!(stage = %job.stage, page = %job.input.page(), error = %failure.error, "retrying stage after memory pressure");
        let _permit = self.accelerator.recover(job.stage, &self.stages).await;
        if job.input.stopped() {
            return Ok(StageOutcome::Stopped);
        }
        match self.load_and_process(job, model).await {
            Ok(outcome) => Ok(outcome),
            Err(failure) => Err(self.stage_error(job.stage, model, failure)),
        }
    }

    async fn load_and_process(
        &self,
        job: &StageJob,
        model: &str,
    ) -> std::result::Result<StageOutcome, AttemptFailure> {
        progress::emit(
            job.progress.as_ref(),
            Progress::Loading {
                page: job.input.page(),
                stage: job.stage,
                model: model.to_owned(),
            },
        );
        self.stages
            .load(job.stage)
            .await
            .map_err(|error| AttemptFailure {
                kind: ErrorKind::ModelLoad,
                error,
            })?;
        if job.input.stopped() {
            return Ok(StageOutcome::Stopped);
        }
        progress::emit(
            job.progress.as_ref(),
            Progress::Running {
                page: job.input.page(),
                stage: job.stage,
                model: model.to_owned(),
            },
        );
        self.stages
            .process(job.stage, job.input.clone())
            .await
            .map(|patch| {
                if patch.is_empty() {
                    StageOutcome::Skipped
                } else {
                    StageOutcome::Patch(patch)
                }
            })
            .map_err(|error| AttemptFailure {
                kind: ErrorKind::Processing,
                error,
            })
    }

    fn stage_error(&self, stage: Stage, model: &str, failure: AttemptFailure) -> PipelineError {
        self.stages.unload(stage);
        let message = match failure.kind {
            ErrorKind::ModelLoad => format!("failed to load {model}"),
            _ => format!("{model} failed"),
        };
        PipelineError::new(failure.kind, Some(stage), failure.error.context(message))
    }
}

struct AttemptFailure {
    kind: ErrorKind,
    error: anyhow::Error,
}

fn is_out_of_memory(error: &anyhow::Error) -> bool {
    error.chain().any(|source| {
        let message = source.to_string().to_ascii_lowercase();
        message.contains("out of memory")
            || message.contains("cuda_error_out_of_memory")
            || message.contains("not enough memory")
    })
}

pub(crate) struct StageJob {
    stage: Stage,
    input: StageInput,
    progress: Option<ProgressSink>,
}

impl StageJob {
    pub(crate) fn new(stage: Stage, input: StageInput, progress: Option<ProgressSink>) -> Self {
        Self {
            stage,
            input,
            progress,
        }
    }
}

pub(crate) enum StageOutcome {
    Patch(Patch),
    Skipped,
    Stopped,
}

pub(crate) struct StageCompletion {
    pub(crate) page: EntityId,
    pub(crate) stage: Stage,
    pub(crate) model: String,
    pub(crate) elapsed: Duration,
    pub(crate) outcome: std::result::Result<StageOutcome, PipelineError>,
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use anyhow::Context as _;
    use koharu_llama::{DecodeError, LlamaContextLoadError, mtmd::MtmdEvalError};

    use super::*;
    use crate::{ImageCache, StopToken, stages::StageProcessor};

    #[test]
    fn llama_cpp_allocation_failures_are_out_of_memory() {
        let context = Err::<(), _>(LlamaContextLoadError::NullReturn)
            .context("failed to create llama.cpp context")
            .unwrap_err();
        let decode = Err::<(), _>(DecodeError::from(std::num::NonZeroI32::new(-2).unwrap()))
            .context("failed to decode prompt batch")
            .unwrap_err();
        let eval = Err::<(), _>(MtmdEvalError::from_code(-2))
            .context("failed to evaluate multimodal prompt")
            .unwrap_err();

        assert!(is_out_of_memory(&context));
        assert!(is_out_of_memory(&decode));
        assert!(is_out_of_memory(&eval));
    }

    #[test]
    fn other_llama_cpp_failures_are_not_out_of_memory() {
        let invalid = anyhow::Error::new(DecodeError::from(std::num::NonZeroI32::new(-1).unwrap()));
        let failed = anyhow::Error::new(DecodeError::from(std::num::NonZeroI32::new(-3).unwrap()));

        assert!(!is_out_of_memory(&invalid));
        assert!(!is_out_of_memory(&failed));
    }

    /// Simulates a processor whose long-running work ends early because the job was stopped.
    struct Interrupted {
        stop: StopToken,
        unloads: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl StageProcessor for Interrupted {
        fn model(&self) -> &'static str {
            "interrupted"
        }

        fn unload(&self) -> bool {
            self.unloads.fetch_add(1, Ordering::SeqCst);
            true
        }

        async fn load(&self) -> Result<()> {
            Ok(())
        }

        async fn process(&self, _input: StageInput) -> Result<Patch> {
            self.stop.stop();
            anyhow::bail!("generation ended before its output was complete")
        }
    }

    #[tokio::test]
    async fn stop_during_processing_keeps_the_model_loaded() {
        let device = koharu_ml::Device::cpu();
        let translator = koharu_translator::Translator::from_config(
            device.clone(),
            koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        )
        .unwrap();
        let mut runner = StageRunner::new(
            &PipelineConfig::default(),
            translator,
            &device,
            ResourceMonitor::new(&device),
        )
        .unwrap();
        let stop = StopToken::default();
        let processor = Arc::new(Interrupted {
            stop: stop.clone(),
            unloads: AtomicUsize::new(0),
        });
        runner.set_test_processor(processor.clone());
        let input = StageInput::new(
            koharu_scene::Session::memory().await.unwrap().snapshot(),
            EntityId::new(),
            None,
            None,
            Arc::new(ImageCache::default()),
            None,
            stop,
        );

        let completion = runner.run(StageJob::new(Stage::Ocr, input, None)).await;

        assert_eq!(
            processor.unloads.load(Ordering::SeqCst),
            0,
            "stopping must not unload the stage model"
        );
        assert!(
            matches!(completion.outcome, Ok(StageOutcome::Stopped)),
            "a stop that interrupts processing is not a stage failure"
        );
    }
}
