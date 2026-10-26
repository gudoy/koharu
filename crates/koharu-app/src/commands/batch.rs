use std::{
    collections::BTreeMap,
    fs,
    io::Write as _,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context as _, Result, bail};
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
const FORMATS: [ExportFormat; 2] = [ExportFormat::Psd, ExportFormat::Png];

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

    let (_, task) = processing::start(
        handle,
        koharu_pipeline::Scope::Project,
        koharu_pipeline::Operation::Full,
    )
    .await?;
    match task.await.context("processing task failed")? {
        (JobState::Stopped, _) => return Ok(Outcome::Stopped),
        (JobState::Failed, error) => bail!(error.unwrap_or_else(|| "processing failed".to_owned())),
        _ => {}
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
    for format in FORMATS {
        output::write(&desktop, snapshot.clone(), format, output.to_owned()).await?;
    }
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
