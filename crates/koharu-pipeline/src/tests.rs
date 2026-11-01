use super::*;

#[test]
fn configuration_ignores_unknown_fields() {
    let config = toml::from_str::<PipelineConfig>("legacy_limit = 1").unwrap();
    assert_eq!(config, PipelineConfig::default());
}

#[tokio::test]
async fn stop_is_a_successful_partial_result() {
    let pipeline = pipeline(Default::default());
    let stop = StopToken::default();
    stop.stop();
    let request = Request {
        stop,
        ..Request::default()
    };
    let mut committer = RejectCommitter;

    let report = pipeline
        .execute(
            koharu_scene::Session::memory().await.unwrap().snapshot(),
            request,
            &mut committer,
        )
        .await
        .unwrap();

    assert_eq!(report.status, RunStatus::Stopped);
    assert_eq!(report.completed, 0);
}

#[tokio::test]
async fn stop_after_a_page_keeps_completed_progress() {
    let translation = TranslationConfig {
        model: koharu_translator::ModelSelection {
            provider: koharu_translator::Provider::OpenAi,
            model: Some("gpt-5.6-luna".to_owned()),
            quantization: None,
            vision: true,
            reasoning: true,
        },
        ..Default::default()
    };
    let pipeline = pipeline(translation);
    let mut session = koharu_scene::Session::memory().await.unwrap();
    let patch = session
        .snapshot()
        .patch(|edit| {
            edit.add_page(
                koharu_scene::PageDraft::new("one", 1.0, 1.0),
                koharu_scene::At::End,
            )?;
            edit.add_page(
                koharu_scene::PageDraft::new("two", 1.0, 1.0),
                koharu_scene::At::End,
            )?;
            Ok(())
        })
        .unwrap();
    session.commit(patch).await.unwrap();
    let stop = StopToken::default();
    let progress_stop = stop.clone();
    let request = Request {
        operation: Operation::Only {
            stage: Stage::Translation,
        },
        stop,
        progress: Some(std::sync::Arc::new(move |event| {
            if matches!(event, Progress::Skipped { .. }) {
                progress_stop.stop();
            }
        })),
        ..Request::default()
    };
    let mut committer = RejectCommitter;

    let report = pipeline
        .execute(session.snapshot(), request, &mut committer)
        .await
        .unwrap();

    assert_eq!(report.status, RunStatus::Stopped);
    assert_eq!(report.completed, 1);
    assert_eq!(report.total, 2);
}

/// Fails every stage on one page and finds nothing to do on the others.
struct FailingPage {
    page: koharu_scene::EntityId,
    snapshot: koharu_scene::Snapshot,
}

#[async_trait::async_trait]
impl stages::StageProcessor for FailingPage {
    fn model(&self) -> &'static str {
        "failing-page"
    }

    fn unload(&self) -> bool {
        false
    }

    async fn load(&self) -> anyhow::Result<()> {
        Ok(())
    }

    async fn process(&self, input: stages::StageInput) -> anyhow::Result<koharu_scene::Patch> {
        anyhow::ensure!(input.page() != self.page, "model returned invalid output");
        Ok(self.snapshot.patch(|_| Ok(()))?)
    }
}

#[tokio::test]
async fn a_page_failure_does_not_stop_the_other_pages() {
    let mut session = koharu_scene::Session::memory().await.unwrap();
    let mut pages = Vec::new();
    let patch = session
        .snapshot()
        .patch(|edit| {
            for name in ["one", "two"] {
                pages.push(edit.add_page(
                    koharu_scene::PageDraft::new(name, 1.0, 1.0),
                    koharu_scene::At::End,
                )?);
            }
            Ok(())
        })
        .unwrap();
    session.commit(patch).await.unwrap();
    let snapshot = session.snapshot();

    let device = koharu_ml::Device::cpu();
    let translator = koharu_translator::Translator::from_config(
        device.clone(),
        koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
    )
    .unwrap();
    let resources = resources::ResourceMonitor::new(&device);
    let mut runner = stage_runner::StageRunner::new(
        &PipelineConfig::default(),
        translator,
        &device,
        resources.clone(),
    )
    .unwrap();
    runner.set_test_processor(std::sync::Arc::new(FailingPage {
        page: pages[0],
        snapshot: snapshot.clone(),
    }));
    let failed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let events = failed.clone();
    let request = Request {
        operation: Operation::Through {
            stage: Stage::Translation,
        },
        progress: Some(std::sync::Arc::new(move |event| {
            if let Progress::Failed {
                page,
                stage,
                abandoned,
            } = event
            {
                events.lock().unwrap().push((page, stage, abandoned));
            }
        })),
        ..Request::default()
    };
    let mut committer = RejectCommitter;

    let report = execution::Execution::new(
        std::sync::Arc::new(runner),
        resources,
        snapshot,
        request,
        &mut committer,
    )
    .unwrap()
    .run()
    .await
    .unwrap();

    assert_eq!(report.status, RunStatus::Completed);
    assert_eq!(report.completed, 3);
    assert_eq!(report.total, 6);
    assert_eq!(report.failures.len(), 1);
    assert_eq!(report.failures[0].page, pages[0]);
    assert_eq!(report.failures[0].error.stage, Some(Stage::Detection));
    assert_eq!(
        *failed.lock().unwrap(),
        [(
            pages[0],
            Stage::Detection,
            vec![Stage::Ocr, Stage::Translation]
        )]
    );
}

struct RejectCommitter;

fn pipeline(translation: TranslationConfig) -> Pipeline {
    let config = PipelineConfig {
        translation,
        ..PipelineConfig::default()
    };
    Pipeline::from_config(
        koharu_config::Config::memory(config),
        koharu_config::Config::memory(koharu_translator::ProvidersConfig::default()),
        koharu_ml::Device::cpu(),
    )
    .unwrap()
}

#[async_trait::async_trait]
impl Committer for RejectCommitter {
    async fn commit(&mut self, _output: StageOutput) -> anyhow::Result<koharu_scene::Snapshot> {
        anyhow::bail!("stopped execution must not commit")
    }
}

#[test]
fn operations_expand_to_the_supported_workflows() {
    assert_eq!(
        Operation::Through {
            stage: Stage::Translation,
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Ocr, Stage::Translation],
    );
    assert_eq!(
        Operation::Through {
            stage: Stage::Inpainting,
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Inpainting],
    );
    assert_eq!(
        Operation::Only {
            stage: Stage::Translation,
        }
        .stages()
        .unwrap(),
        vec![Stage::Translation],
    );
    assert_eq!(
        Operation::Stages {
            stages: vec![Stage::Translation, Stage::Detection, Stage::Translation],
        }
        .stages()
        .unwrap(),
        vec![Stage::Detection, Stage::Translation],
    );
}
