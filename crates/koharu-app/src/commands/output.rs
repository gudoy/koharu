use anyhow::{Context as _, Result};
use futures::{StreamExt as _, TryStreamExt as _, stream};
use image::{
    ExtendedColorType, ImageEncoder as _, RgbaImage,
    codecs::png::{CompressionType, FilterType, PngEncoder},
};
use koharu_psd::{PsdExportOptions, export_page};
use koharu_rasterizer::{Raster, RasterOptions, Rasterizer};
use koharu_renderer::{Frame, Renderer};
use koharu_scene::{AssetRole, EntityId, Snapshot};
use serde::Deserialize;
use specta::Type;
use std::{io::Write as _, path::PathBuf, sync::Arc};
use tauri::{State, WebviewWindow, ipc::IpcResponse};
use tauri_runtime_cef::CefRuntime;

use super::{Error, project::CurrentProject};
use koharu_desktop::Desktop;

const THUMBNAIL_EDGE: u32 = 128;

#[derive(Type)]
#[specta(transparent)]
pub(crate) struct ThumbnailBytes(#[specta(type = Vec<u8>)] Vec<u8>);

impl IpcResponse for ThumbnailBytes {
    fn body(self) -> tauri::Result<tauri::ipc::InvokeResponseBody> {
        Ok(self.0.into())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Type)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Png,
    Psd,
    Cbz,
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn export(
    window: WebviewWindow<CefRuntime>,
    format: ExportFormat,
    project: State<'_, CurrentProject>,
    desktop: State<'_, Desktop>,
) -> std::result::Result<(), Error> {
    let (name, snapshot) = {
        let project = project.project.lock().await;
        let project = project.as_ref().context("no project is open")?;
        (project.name.clone(), project.snapshot())
    };
    if snapshot.pages().next().is_none() {
        return Err(anyhow::anyhow!("there are no pages to export").into());
    }
    let dialog = rfd::AsyncFileDialog::new().set_parent(&window);
    let destination = match format {
        ExportFormat::Png | ExportFormat::Psd => dialog.pick_folder().await,
        ExportFormat::Cbz => {
            dialog
                .add_filter("Comic Book Archive", &["cbz"])
                .set_file_name(format!("{name}.cbz"))
                .save_file()
                .await
        }
    };
    let Some(destination) = destination.map(|destination| destination.path().to_owned()) else {
        return Ok(());
    };
    write(&desktop, snapshot, &[format], destination).await?;
    Ok(())
}

/// Pages prepared at once during an export. Each one holds a full-resolution
/// frame and its encoded file, so preparing a whole long project together
/// exhausts memory.
const EXPORT_CONCURRENCY: usize = 4;

/// Renders every page of `snapshot` once and writes it to `destination` in
/// each of `formats`: a folder for PNG and PSD, an archive path for CBZ.
pub(crate) async fn write(
    desktop: &Desktop,
    snapshot: Snapshot,
    formats: &[ExportFormat],
    destination: PathBuf,
) -> Result<()> {
    let pages = snapshot.pages().map(|page| page.id()).collect::<Vec<_>>();
    if pages.is_empty() {
        anyhow::bail!("there are no pages to export");
    }
    let is_archive = formats.contains(&ExportFormat::Cbz);
    if is_archive && formats.len() != 1 {
        anyhow::bail!("a CBZ archive cannot be exported together with another format");
    }
    let renderer = desktop.renderer();
    let rasterizer = desktop.rasterizer().await?;
    let options = PsdExportOptions::default();
    let mut archive = if is_archive {
        let directory = destination.parent().context("archive path has no parent")?;
        Some(zip::ZipWriter::new(tempfile::NamedTempFile::new_in(
            directory,
        )?))
    } else {
        None
    };
    let width = pages.len().to_string().len().max(4);
    // `buffered` yields in project order while bounding the pages in flight.
    let mut encoded = stream::iter(pages.iter().copied())
        .map(|page| {
            let (renderer, snapshot, options) = (&renderer, &snapshot, &options);
            let rasterizer = Arc::clone(&rasterizer);
            async move {
                // One prepared frame and its composite feed every format.
                let frame = renderer.render(snapshot, page).await?;
                let mut merged = Some(
                    rasterize(Arc::clone(&rasterizer), &frame, RasterOptions::default())
                        .await?
                        .image,
                );
                let mut files = Vec::with_capacity(formats.len());
                for (index, format) in formats.iter().enumerate() {
                    let composite = if index + 1 < formats.len() {
                        merged.clone()
                    } else {
                        merged.take()
                    }
                    .expect("only the last format takes the composite");
                    files.push(match format {
                        ExportFormat::Png | ExportFormat::Cbz => (
                            "png",
                            tokio_rayon::spawn(move || encode_png(&composite)).await?,
                        ),
                        ExportFormat::Psd => (
                            "psd",
                            export_page(
                                Arc::clone(&rasterizer),
                                snapshot,
                                &frame,
                                composite,
                                options,
                            )
                            .await?,
                        ),
                    });
                }
                anyhow::Ok(files)
            }
        })
        .buffered(EXPORT_CONCURRENCY);
    let mut index = 0;
    while let Some(files) = encoded.try_next().await? {
        let page = snapshot.page(pages[index])?.page()?;
        let name = page
            .label
            .trim()
            .trim_end_matches(|character: char| character == '.' || character.is_whitespace());
        let name = name
            .rsplit_once('.')
            .map_or(name, |(stem, _)| stem)
            .replace(['<', '>', ':', '"', '/', '\\', '|', '?', '*'], "_");
        for (extension, bytes) in files {
            let name = format!(
                "{:0width$}_{}.{extension}",
                index + 1,
                if name.is_empty() { "page" } else { &name }
            );
            if let Some(mut writer) = archive.take() {
                // Archive I/O is synchronous, so it runs off the async workers.
                archive = Some(
                    tokio::task::spawn_blocking(move || -> Result<_> {
                        // PNG data is already compressed.
                        let options = zip::write::SimpleFileOptions::default()
                            .compression_method(zip::CompressionMethod::Stored);
                        writer.start_file(name, options)?;
                        writer.write_all(&bytes)?;
                        Ok(writer)
                    })
                    .await??,
                );
            } else {
                tokio::fs::write(destination.join(name), bytes).await?;
            }
        }
        index += 1;
    }
    if let Some(writer) = archive {
        tokio::task::spawn_blocking(move || -> Result<()> {
            writer.finish()?.persist(destination)?;
            Ok(())
        })
        .await??;
    }
    Ok(())
}

fn encode_png(image: &RgbaImage) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    // `Best` costs about 2.5x the encode time of `Default` for under 2% smaller pages.
    PngEncoder::new_with_quality(&mut bytes, CompressionType::Default, FilterType::Adaptive)
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            ExtendedColorType::Rgba8,
        )?;
    Ok(bytes)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_thumbnail(
    page: EntityId,
    project: State<'_, CurrentProject>,
) -> std::result::Result<ThumbnailBytes, Error> {
    let snapshot = project
        .project
        .lock()
        .await
        .as_ref()
        .context("no project is open")?
        .snapshot();
    snapshot.page(page)?;
    let blob = snapshot
        .asset(page, &AssetRole::new("source")?)?
        .with_context(|| format!("page {page} has no source image"))?
        .blob;
    let bytes = snapshot.read_blob(blob).await?;
    let bytes = tokio_rayon::spawn(move || -> Result<Vec<u8>> {
        let image = image::load_from_memory(&bytes).context("failed to decode source image")?;
        if image.width() == 0 || image.height() == 0 {
            return Err(anyhow::anyhow!("source image is empty"));
        }
        let image = image.thumbnail(THUMBNAIL_EDGE, THUMBNAIL_EDGE).to_rgba8();
        let encoder = webp::Encoder::from_rgba(image.as_raw(), image.width(), image.height());
        Ok(encoder.encode(80.0).to_vec())
    })
    .await?;
    Ok(ThumbnailBytes(bytes))
}

pub(crate) async fn rendered_preview(
    renderer: &Renderer,
    rasterizer: Arc<Rasterizer>,
    snapshot: &Snapshot,
    page: EntityId,
) -> Result<Vec<u8>> {
    snapshot.page(page)?;
    let frame = renderer.render(snapshot, page).await?;
    let image = rasterize(rasterizer, &frame, RasterOptions::default())
        .await?
        .image;
    tokio_rayon::spawn(move || {
        let image = image::DynamicImage::ImageRgba8(image)
            .resize(1024, 1024, image::imageops::FilterType::Lanczos3)
            .to_rgba8();
        let encoder = webp::Encoder::from_rgba(image.as_raw(), image.width(), image.height());
        Ok::<_, anyhow::Error>(encoder.encode(85.0).to_vec())
    })
    .await
}

async fn rasterize(
    rasterizer: Arc<Rasterizer>,
    frame: &Frame,
    options: RasterOptions,
) -> Result<Raster> {
    let frame = frame.raster_frame()?;
    tokio_rayon::spawn(move || rasterizer.rasterize(&frame, options))
        .await
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        io::Cursor,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use koharu_scene::{AssetInput, AssetMetadata, At, PageDraft, Session};

    use super::*;

    /// Counts the `Renderer::render` spans opened on the current thread.
    struct Renders(Arc<AtomicUsize>);

    impl tracing::Subscriber for Renders {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            let metadata = span.metadata();
            if metadata.name() == "render" && metadata.target() == "koharu_renderer::renderer" {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
            tracing::span::Id::from_u64(1)
        }

        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}

        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

        fn event(&self, _: &tracing::Event<'_>) {}

        fn enter(&self, _: &tracing::span::Id) {}

        fn exit(&self, _: &tracing::span::Id) {}
    }

    #[tokio::test]
    async fn batch_formats_render_each_page_once() {
        let mut session = Session::memory().await.unwrap();
        let source = AssetRole::new("source").unwrap();
        let mut png = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            4,
            4,
            image::Rgba([10, 20, 200, 255]),
        ))
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
        let png: Arc<[u8]> = png.into_inner().into();
        let patch = session
            .snapshot()
            .patch(|edit| {
                for label in ["one", "two"] {
                    let page = edit.add_page(PageDraft::new(label, 4.0, 4.0), At::End)?;
                    edit.set_asset(
                        page,
                        &source,
                        AssetInput::new(
                            png.clone(),
                            "image/png",
                            AssetMetadata {
                                width: Some(4),
                                height: Some(4),
                                attributes: BTreeMap::new(),
                            },
                        ),
                    )?;
                }
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let desktop = Desktop::new().unwrap();
        let destination = tempfile::tempdir().unwrap();
        let renders = Arc::new(AtomicUsize::new(0));
        let _counting = tracing::subscriber::set_default(Renders(Arc::clone(&renders)));

        write(
            &desktop,
            snapshot,
            &crate::commands::batch::FORMATS,
            destination.path().to_owned(),
        )
        .await
        .unwrap();

        assert_eq!(
            renders.load(Ordering::Relaxed),
            2,
            "each page renders once for both formats"
        );
        let mut names = std::fs::read_dir(destination.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(
            names,
            [
                "0001_one.png",
                "0001_one.psd",
                "0002_two.png",
                "0002_two.psd"
            ]
        );
    }
}
