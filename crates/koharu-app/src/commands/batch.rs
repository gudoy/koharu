use std::{
    collections::BTreeMap,
    fs,
    io::Write as _,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context as _, Result, bail};
use koharu_scene::{
    AssetRole, EntityId, LanguageTag, Origin, RasterLayer, RasterLayerKind, RecognizedFrom, Region,
    RegionSpec as _, Snapshot, SourceText, TextRegion,
};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager as _, WebviewWindow};
use tauri_runtime_cef::CefRuntime;

use super::{
    Error,
    lifecycle::{folder_files, import_files, replace_project},
    output::{self, ExportFormat},
    processing::{self, JobState, Processing},
    project::{CurrentProject, ProjectLibrary},
};
use koharu_desktop::Desktop;

/// Written beside the exported folders. It records each folder's outcome so a
/// rerun resumes after the last completed one and failures can be inspected.
const REPORT_FILE: &str = "koharu-batch.json";
pub(crate) const FORMATS: [ExportFormat; 2] = [ExportFormat::Psd, ExportFormat::Png];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Running,
    Done,
    Failed,
    Stopped,
}

#[derive(Debug, Deserialize, Serialize)]
struct Entry {
    status: Status,
    pages: Option<usize>,
    error: Option<String>,
    updated: u64,
}

/// Source and destination of a batch started from the command line.
pub struct Batch {
    pub source: std::path::PathBuf,
    pub destination: std::path::PathBuf,
}

enum Outcome {
    Done(usize),
    Stopped,
}

/// Translates every subfolder of a chosen source folder as its own project and
/// exports each one as PSD and PNG into a same-named folder of the destination.
#[tauri::command]
#[specta::specta]
pub(crate) async fn batch(
    window: WebviewWindow<CefRuntime>,
    handle: AppHandle<CefRuntime>,
) -> std::result::Result<(), Error> {
    if !handle.state::<Processing>().stops.lock().is_empty() {
        return Err(anyhow::anyhow!("a batch cannot start while processing is running").into());
    }
    let Some(source) = rfd::AsyncFileDialog::new()
        .set_title("Folder containing one folder per project")
        .set_parent(&window)
        .pick_folder()
        .await
    else {
        return Ok(());
    };
    let Some(destination) = rfd::AsyncFileDialog::new()
        .set_title("Export destination")
        .set_parent(&window)
        .pick_folder()
        .await
    else {
        return Ok(());
    };
    run(&handle, source.path(), destination.path()).await?;
    Ok(())
}

pub(crate) async fn run(
    handle: &AppHandle<CefRuntime>,
    source: &Path,
    destination: &Path,
) -> Result<()> {
    // Exports would otherwise land inside the folders being imported.
    if destination == source {
        bail!("the export destination must differ from the source folder");
    }
    let report_path = destination.join(REPORT_FILE);
    let mut report = load_report(&report_path)?;
    let mut folders = fs::read_dir(source)
        .with_context(|| format!("failed to read {}", source.display()))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .map(|entry| entry.path())
        .filter(|path| {
            path != destination
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| !name.starts_with('.'))
        })
        .collect::<Vec<_>>();
    alphanumeric_sort::sort_path_slice(&mut folders);
    tracing::info!(
        source = %source.display(),
        destination = %destination.display(),
        folders = folders.len(),
        "batch started"
    );

    for folder in folders {
        let name = folder
            .file_name()
            .and_then(|name| name.to_str())
            .context("folder name is not valid UTF-8")?
            .to_owned();
        if report
            .get(&name)
            .is_some_and(|entry| entry.status == Status::Done)
        {
            continue;
        }
        update(
            &mut report,
            &report_path,
            &name,
            Status::Running,
            None,
            None,
        )?;
        tracing::info!(project = %name, "batch project started");
        match translate(handle, &folder, &name, &destination.join(&name)).await {
            Ok(Outcome::Done(pages)) => {
                tracing::info!(project = %name, pages, "batch project exported");
                update(
                    &mut report,
                    &report_path,
                    &name,
                    Status::Done,
                    Some(pages),
                    None,
                )?;
            }
            Ok(Outcome::Stopped) => {
                tracing::info!(project = %name, "batch stopped");
                update(
                    &mut report,
                    &report_path,
                    &name,
                    Status::Stopped,
                    None,
                    None,
                )?;
                return Ok(());
            }
            Err(error) => {
                let error = format!("{error:#}");
                tracing::error!(project = %name, %error, "batch project failed");
                update(
                    &mut report,
                    &report_path,
                    &name,
                    Status::Failed,
                    None,
                    Some(error),
                )?;
            }
        }
    }
    tracing::info!("batch finished");
    Ok(())
}

async fn translate(
    handle: &AppHandle<CefRuntime>,
    folder: &Path,
    name: &str,
    output: &Path,
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
    if !pending.is_empty() {
        let (_, task) = processing::start(
            handle,
            koharu_pipeline::Scope::Pages(pending),
            koharu_pipeline::Operation::Full,
        )
        .await?;
        match task.await.context("processing task failed")? {
            (JobState::Stopped, _) => return Ok(Outcome::Stopped),
            (JobState::Failed, error) => {
                bail!(error.unwrap_or_else(|| "processing failed".to_owned()))
            }
            _ => {}
        }
    }

    let snapshot = handle
        .state::<CurrentProject>()
        .project
        .lock()
        .await
        .as_ref()
        .context("no project is open")?
        .snapshot();
    let pages = snapshot.pages().count();
    fs::create_dir_all(output).with_context(|| format!("failed to create {}", output.display()))?;
    let desktop = handle.state::<Desktop>();
    output::write(&desktop, snapshot, &FORMATS, output.to_owned()).await?;
    Ok(Outcome::Done(pages))
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
    status: Status,
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
/// counts because no stage rewrites it. A page without detected text cannot be
/// told apart from an unprocessed one, so it is not finished.
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
            if content
                .source()?
                .is_none_or(|source| source.text.value.trim().is_empty())
            {
                continue;
            }
            let translated = content.translation()?.is_some_and(|translation| {
                matches!(translation.text.origin, Origin::User)
                    || translation.language.as_ref() == Some(target)
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
        let not_cleaned = add_page(&mut edit, false);
        let no_text = add_page(&mut edit, true);
        let done_text = add_text(&mut edit, done);
        let untranslated_text = add_text(&mut edit, untranslated);
        let other_language_text = add_text(&mut edit, other_language);
        add_text(&mut edit, unread);
        let blank_text = add_text(&mut edit, blank);
        let edited_text = add_text(&mut edit, edited);
        let not_cleaned_text = add_text(&mut edit, not_cleaned);
        // Neither OCR nor translation rewrites user-authored text.
        edit.set(edited_text, &source("こんにちは")).unwrap();
        edit.set(edited_text, &translation("Hi", None)).unwrap();
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
        for content in [done_text, not_cleaned_text] {
            edit.set(content, &translation("Hello", Some("en")))
                .unwrap();
        }
        edit.set(other_language_text, &translation("Bonjour", Some("fr")))
            .unwrap();
        session.commit(edit.finish().unwrap()).await.unwrap();

        assert_eq!(
            pending_pages(&session.snapshot(), &LanguageTag::new("en").unwrap()).unwrap(),
            [untranslated, other_language, unread, not_cleaned, no_text]
        );
    }
}
