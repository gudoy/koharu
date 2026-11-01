use std::sync::Arc;

use anyhow::{Context as _, Result};
use arc_swap::ArcSwap;
use koharu_config::Config;
use koharu_scene::Snapshot;

use crate::{
    Committer, PipelineConfig, PipelineError, Report, Request, ResourceSnapshot,
    execution::Execution, resources::ResourceMonitor, stage_runner::StageRunner,
};

#[derive(Clone)]
pub struct Pipeline {
    current: Arc<ArcSwap<StageRunner>>,
    resources: Arc<ResourceMonitor>,
    execution: Arc<tokio::sync::Mutex<()>>,
}

impl Pipeline {
    pub fn load(device: koharu_ml::Device) -> Result<Self> {
        Self::from_config(
            PipelineConfig::load()?,
            koharu_translator::ProvidersConfig::load()?,
            device,
        )
    }

    #[tracing::instrument(skip_all)]
    pub fn from_config(
        config: Config<PipelineConfig>,
        providers: Config<koharu_translator::ProvidersConfig>,
        device: koharu_ml::Device,
    ) -> Result<Self> {
        let translator = koharu_translator::Translator::from_config(device.clone(), providers)?;
        let resources = ResourceMonitor::new(&device);
        let runner = {
            let value = config.read()?;
            StageRunner::new(&value, translator.clone(), &device, resources.clone())?
        };
        let current = Arc::new(ArcSwap::from_pointee(runner));
        let watched = current.clone();
        let _watcher = tokio::runtime::Handle::try_current()
            .context("pipeline requires a Tokio runtime")?
            .spawn(async move {
                let mut changes = config.subscribe();
                while changes.changed().await.is_ok() {
                    let runner = config.read().and_then(|value| {
                        watched
                            .load()
                            .reconfigure(&value, translator.clone(), &device)
                    });
                    match runner {
                        Ok(runner) => watched.store(Arc::new(runner)),
                        Err(error) => tracing::error!(%error, "failed to reload pipeline"),
                    }
                }
            });
        Ok(Self {
            current,
            resources,
            execution: Arc::new(tokio::sync::Mutex::new(())),
        })
    }

    pub fn subscribe_resources(&self) -> tokio::sync::watch::Receiver<ResourceSnapshot> {
        self.resources.start();
        self.resources.subscribe()
    }

    #[tracing::instrument(skip_all)]
    pub async fn execute(
        &self,
        snapshot: Snapshot,
        request: Request,
        committer: &mut dyn Committer,
    ) -> std::result::Result<Report, PipelineError> {
        let _execution = self.execution.lock().await;
        Execution::new(
            self.current.load_full(),
            self.resources.clone(),
            snapshot,
            request,
            committer,
        )?
        .run()
        .await
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::{OcrModel, Stage};

    #[tokio::test]
    async fn reload_keeps_processors_whose_stage_config_is_unchanged() {
        let config = Config::memory(PipelineConfig::default());
        let pipeline = Pipeline::from_config(
            config.clone(),
            Config::memory(koharu_translator::ProvidersConfig::default()),
            koharu_ml::Device::cpu(),
        )
        .unwrap();
        // Let the watcher subscribe before the first write.
        tokio::task::yield_now().await;

        let before = pipeline.current.load_full();
        config.write().unwrap().translation.target_language = koharu_translator::Language::French;
        let translated = reloaded(&pipeline, &before).await;
        for stage in [Stage::Detection, Stage::Ocr, Stage::Inpainting] {
            assert!(
                translated.shares_processor(&before, stage),
                "{stage} processor was rebuilt for a translation-only change"
            );
        }

        config.write().unwrap().ocr = OcrModel::MangaOcr;
        let recognized = reloaded(&pipeline, &translated).await;
        assert!(!recognized.shares_processor(&translated, Stage::Ocr));
        assert!(recognized.shares_processor(&translated, Stage::Detection));
        assert!(recognized.shares_processor(&translated, Stage::Inpainting));
    }

    async fn reloaded(pipeline: &Pipeline, previous: &Arc<StageRunner>) -> Arc<StageRunner> {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let current = pipeline.current.load_full();
                if !Arc::ptr_eq(&current, previous) {
                    return current;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("pipeline reloads after a configuration change")
    }
}
