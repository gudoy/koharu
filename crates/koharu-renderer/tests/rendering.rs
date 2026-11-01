use std::{
    collections::{BTreeMap, HashMap},
    io::Cursor,
    sync::Arc,
};

use koharu_rasterizer::{PREPARED_RASTER_TILE_DIMENSION, RasterOptions, Rasterizer, ResourceId};
use koharu_renderer::{ImageKind, LayerKind, Renderer};
use koharu_scene::{
    AssetInput, AssetMetadata, AssetRole, At, Geometry, Origin, PageDraft, RasterLayer,
    RasterLayerKind, Session,
};

fn png(width: u32, height: u32, color: [u8; 4]) -> Arc<[u8]> {
    let image = image::RgbaImage::from_pixel(width, height, image::Rgba(color));
    let mut bytes = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(image)
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    bytes.into_inner().into()
}

fn png_image(image: image::RgbaImage) -> Arc<[u8]> {
    let mut bytes = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(image)
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    bytes.into_inner().into()
}

#[tokio::test]
async fn public_renderer_returns_a_complete_immutable_frame() {
    let mut session = Session::memory().await.unwrap();
    let mut page = None;
    let patch = session
        .snapshot()
        .patch(|edit| {
            page = Some(edit.add_page(PageDraft::new("page", 320.0, 240.0), At::End)?);
            Ok(())
        })
        .unwrap();
    let snapshot = session.commit(patch).await.unwrap().snapshot;
    let page = page.unwrap();

    let renderer = Renderer::new().unwrap();
    let frame = renderer.render(&snapshot, page).await.unwrap();

    assert_eq!(frame.page(), page);
    assert_eq!(frame.revision(), snapshot.revision());
    assert_eq!(frame.size(), (320, 240));
    assert_eq!(frame.origin(), (0, 0));
    assert!(frame.layers().is_empty());
    assert!(frame.layer(page).is_none());
    assert!(frame
        .diagnostics()
        .iter()
        .any(|diagnostic| matches!(diagnostic, koharu_renderer::RenderDiagnostic::MissingAsset { entity, role } if *entity == page && role == "source")));
}

#[tokio::test]
async fn native_rasterization_preserves_pixels_across_prepared_tile_edges() {
    let width = PREPARED_RASTER_TILE_DIMENSION + 1;
    let red = [230, 20, 30, 255];
    let blue = [10, 40, 220, 255];
    let mut source_image = image::RgbaImage::from_pixel(width, 1, image::Rgba(red));
    source_image.put_pixel(width - 1, 0, image::Rgba(blue));
    let source = AssetRole::new("source").unwrap();
    let mut session = Session::memory().await.unwrap();
    let mut page = None;
    let patch = session
        .snapshot()
        .patch(|edit| {
            let created = edit.add_page(PageDraft::new("page", width as f64, 1.0), At::End)?;
            edit.set_asset(
                created,
                &source,
                AssetInput::new(
                    png_image(source_image),
                    "image/png",
                    AssetMetadata {
                        width: Some(width),
                        height: Some(1),
                        attributes: BTreeMap::new(),
                    },
                ),
            )?;
            page = Some(created);
            Ok(())
        })
        .unwrap();
    let snapshot = session.commit(patch).await.unwrap().snapshot;
    let renderer = Renderer::new().unwrap();
    let frame = renderer.render(&snapshot, page.unwrap()).await.unwrap();
    let manifest = frame.prepared().manifest().unwrap();
    assert_eq!(manifest.resources.len(), 1);
    let koharu_rasterizer::PreparedContent::Raster(prepared) = &manifest.frame.layers[0].content
    else {
        panic!("source layer must remain raster");
    };
    assert_eq!(prepared.tiles.len(), 2);

    let frame = renderer.decode_rasters(&frame).await.unwrap();
    let raster = Rasterizer::new()
        .unwrap()
        .rasterize(&frame.raster_frame().unwrap(), RasterOptions::default())
        .unwrap();

    assert_eq!(raster.image.get_pixel(width - 2, 0).0, red);
    assert_eq!(raster.image.get_pixel(width - 1, 0).0, blue);
}

#[tokio::test]
async fn discarded_image_nodes_are_rebuilt_for_a_reopened_presentation() {
    let mut session = Session::memory().await.unwrap();
    let source = AssetRole::new("source").unwrap();
    let color = [12, 34, 56, 255];
    let mut page = None;
    let patch = session
        .snapshot()
        .patch(|edit| {
            let created = edit.add_page(PageDraft::new("page", 4.0, 4.0), At::End)?;
            edit.set_asset(
                created,
                &source,
                AssetInput::new(
                    png(4, 4, color),
                    "image/png",
                    AssetMetadata {
                        width: Some(4),
                        height: Some(4),
                        attributes: BTreeMap::new(),
                    },
                ),
            )?;
            page = Some(created);
            Ok(())
        })
        .unwrap();
    let snapshot = session.commit(patch).await.unwrap().snapshot;
    let page = page.unwrap();
    let renderer = Renderer::new().unwrap();

    renderer.render(&snapshot, page).await.unwrap();
    renderer.discard_retained_nodes();
    let reopened = renderer.render(&snapshot, page).await.unwrap();
    let decoded = renderer.decode_rasters(&reopened).await.unwrap();
    let raster = Rasterizer::new()
        .unwrap()
        .rasterize(&decoded.raster_frame().unwrap(), RasterOptions::default())
        .unwrap();

    assert_eq!(reopened.stats().rebuilt_layers, 1);
    assert!(raster.image.pixels().all(|pixel| pixel.0 == color));
}

#[tokio::test]
async fn rasterized_png_contains_source_and_cleanup_pixels() {
    let mut session = Session::memory().await.unwrap();
    let source = AssetRole::new("source").unwrap();
    let mut cleanup = image::RgbaImage::new(4, 4);
    cleanup.put_pixel(2, 1, image::Rgba([240, 30, 20, 255]));
    let mut page = None;
    let patch = session
        .snapshot()
        .patch(|edit| {
            let created = edit.add_page(PageDraft::new("page", 4.0, 4.0), At::End)?;
            edit.set_asset(
                created,
                &source,
                AssetInput::new(
                    png(4, 4, [10, 20, 200, 255]),
                    "image/png",
                    AssetMetadata {
                        width: Some(4),
                        height: Some(4),
                        attributes: BTreeMap::new(),
                    },
                ),
            )?;
            let cleanup_layer = edit.add_entity(created, At::Start)?;
            edit.set(
                cleanup_layer,
                &RasterLayer {
                    origin: Origin::User,
                    name: "Cleanup".to_owned(),
                    kind: RasterLayerKind::Cleanup,
                },
            )?;
            edit.set_asset(
                cleanup_layer,
                &source,
                AssetInput::new(
                    png_image(cleanup),
                    "image/png",
                    AssetMetadata {
                        width: Some(4),
                        height: Some(4),
                        attributes: BTreeMap::new(),
                    },
                ),
            )?;
            page = Some(created);
            Ok(())
        })
        .unwrap();
    let snapshot = session.commit(patch).await.unwrap().snapshot;
    let renderer = Renderer::new().unwrap();
    let frame = renderer.render(&snapshot, page.unwrap()).await.unwrap();
    let decoded = renderer.decode_rasters(&frame).await.unwrap();
    let raster = Rasterizer::new()
        .unwrap()
        .rasterize(&decoded.raster_frame().unwrap(), RasterOptions::default())
        .unwrap();

    assert!(frame.layers().iter().any(|layer| matches!(
        layer.kind(),
        LayerKind::Image(metadata) if metadata.kind == ImageKind::Source
    )));
    assert!(frame.layers().iter().any(|layer| matches!(
        layer.kind(),
        LayerKind::Image(metadata) if metadata.kind == ImageKind::Cleanup
    )));
    assert_eq!(raster.image.get_pixel(0, 0).0, [10, 20, 200, 255]);
    assert_eq!(raster.image.get_pixel(2, 1).0, [240, 30, 20, 255]);
}

#[tokio::test]
async fn decoded_rasters_match_an_eager_decode_of_the_same_prepared_frame() {
    let mut session = Session::memory().await.unwrap();
    let source = AssetRole::new("source").unwrap();
    let page_image = image::RgbaImage::from_fn(96, 64, |x, y| {
        image::Rgba([(x * 2) as u8, (y * 3) as u8, (x ^ y) as u8, 255])
    });
    let cleanup_image = image::RgbaImage::from_fn(96, 64, |x, y| {
        if (x / 8 + y / 8) % 3 == 0 {
            image::Rgba([250, 250, 250, 255])
        } else {
            image::Rgba([200, 10, 10, ((x + y) % 128) as u8])
        }
    });
    let embedded_image = image::RgbaImage::from_fn(12, 10, |x, y| {
        image::Rgba([(x * 20) as u8, (y * 25) as u8, 99, 180])
    });
    let input = |bytes, width, height| {
        AssetInput::new(
            bytes,
            "image/png",
            AssetMetadata {
                width: Some(width),
                height: Some(height),
                attributes: BTreeMap::new(),
            },
        )
    };
    let mut ids = None;
    let patch = session
        .snapshot()
        .patch(|edit| {
            let page = edit.add_page(PageDraft::new("page", 96.0, 64.0), At::End)?;
            edit.set_asset(page, &source, input(png_image(page_image), 96, 64))?;
            let cleanup = edit.add_entity(page, At::End)?;
            edit.set(
                cleanup,
                &RasterLayer {
                    origin: Origin::User,
                    name: "Cleanup".to_owned(),
                    kind: RasterLayerKind::Cleanup,
                },
            )?;
            edit.set_asset(cleanup, &source, input(png_image(cleanup_image), 96, 64))?;
            let embedded = edit.add_entity(page, At::End)?;
            edit.set(embedded, &Geometry::rectangle(20.5, 11.0, 24.0, 20.0))?;
            edit.set_asset(embedded, &source, input(png_image(embedded_image), 12, 10))?;
            ids = Some((page, embedded));
            Ok(())
        })
        .unwrap();
    let snapshot = session.commit(patch).await.unwrap().snapshot;
    let (page, embedded) = ids.unwrap();
    let renderer = Renderer::new().unwrap();
    let frame = renderer.render(&snapshot, page).await.unwrap();
    assert!(
        frame
            .layers()
            .iter()
            .filter_map(|layer| layer.raster_image())
            .all(|image| image.pixels().is_none())
    );
    assert!(frame.raster_frame().is_err());

    // The eager path decodes every blob up front and feeds the same prepared frame.
    let mut eager = HashMap::new();
    for image in frame
        .layers()
        .iter()
        .filter_map(|layer| layer.raster_image())
    {
        let bytes = snapshot.read_blob(image.blob()).await.unwrap();
        let (width, height) = image.size();
        let pixels = image::load_from_memory(&bytes).unwrap().into_rgba8();
        eager.insert(
            ResourceId::for_encoded_raster(width, height, "image/png", &bytes),
            Arc::<[u8]>::from(pixels.into_raw()),
        );
    }
    let decoded = renderer.decode_rasters(&frame).await.unwrap();
    assert!(std::ptr::eq(frame.prepared(), decoded.prepared()));
    let rasterizer = Rasterizer::new().unwrap();
    let rasterize = |frame: &koharu_rasterizer::Frame| {
        rasterizer
            .rasterize(frame, RasterOptions::default())
            .unwrap()
    };

    let lazy = rasterize(&decoded.raster_frame().unwrap());
    let reference = rasterize(
        &frame
            .prepared()
            .clone()
            .into_frame_with_raster_sources(&eager)
            .unwrap(),
    );
    assert_eq!((lazy.left, lazy.top), (reference.left, reference.top));
    assert!(lazy.image == reference.image);

    let lazy = rasterize(
        &decoded
            .cropped(embedded)
            .unwrap()
            .unwrap()
            .raster_frame()
            .unwrap(),
    );
    let reference = rasterize(
        &frame
            .cropped(embedded)
            .unwrap()
            .unwrap()
            .prepared()
            .clone()
            .into_frame_with_raster_sources(&eager)
            .unwrap(),
    );
    assert_eq!((lazy.left, lazy.top), (reference.left, reference.top));
    assert!(lazy.image == reference.image);
}

#[test]
fn public_layer_kinds_remain_export_metadata() {
    let _ = LayerKind::Image(koharu_renderer::ImageMetadata {
        name: None,
        kind: ImageKind::Source,
    });
}
