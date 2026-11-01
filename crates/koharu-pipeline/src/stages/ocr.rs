use std::sync::{Arc, Mutex};

use super::{StageInput, StageProcessor, finish, generation};
use crate::{ModelCell, OcrModel, StopToken, scope::geometry_extents};
use anyhow::{Context as _, Result, anyhow, bail};
use async_trait::async_trait;
use image::DynamicImage;
use koharu_ml::{
    baberu_ocr::BaberuOcr, hayai_ocr::HayaiOcr, manga_ocr::MangaOcr,
    paddle_ocr_vl::PaddleOCRVLTask, paddle_ocr_vl_quantized::PaddleOCRVLQuantized,
};
use koharu_scene::{
    Authored, EntityId, Geometry, LanguageTag, OcrAnalysis, Origin, RecognizedFrom, Region,
    RegionSpec, SourceText, TextDirection, TextRegion,
};

const PRODUCER: &str = "dev.koharu.pipeline.ocr";

pub(super) struct Processor {
    pub(super) config: OcrModel,
    device: koharu_ml::Device,
    model: ModelCell<Model>,
}

impl Processor {
    pub(super) fn new(config: OcrModel, device: koharu_ml::Device) -> Self {
        Self {
            config,
            device,
            model: ModelCell::new(),
        }
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        match self.config {
            OcrModel::MangaOcr => "manga-ocr",
            OcrModel::BaberuOcr => "baberu-ocr",
            OcrModel::HayaiOcr => "hayai-ocr",
            OcrModel::PaddleOcrVl1_6 => "paddleocr-vl-1.6",
        }
    }

    fn unload(&self) -> bool {
        self.model.unload()
    }

    async fn load(&self) -> Result<()> {
        self.model
            .ensure(|| Model::load(self.device.clone(), &self.config))
            .await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        self.model
            .lock()
            .await
            .as_ref()
            .ok_or_else(|| anyhow!("OCR model is not loaded"))?
            .run(input)
            .await
    }
}

enum Model {
    Manga(Arc<Mutex<MangaOcr>>),
    Baberu(Arc<Mutex<BaberuOcr>>),
    Hayai(Arc<Mutex<HayaiOcr>>),
    Paddle(Arc<Mutex<PaddleOCRVLQuantized>>),
}

impl Model {
    async fn load(device: koharu_ml::Device, config: &OcrModel) -> Result<Self> {
        match config {
            OcrModel::MangaOcr => Ok(Self::Manga(Arc::new(Mutex::new(
                MangaOcr::load(device).await?,
            )))),
            OcrModel::BaberuOcr => Ok(Self::Baberu(Arc::new(Mutex::new(
                BaberuOcr::load(device).await?,
            )))),
            OcrModel::HayaiOcr => Ok(Self::Hayai(Arc::new(Mutex::new(
                HayaiOcr::load(device).await?,
            )))),
            OcrModel::PaddleOcrVl1_6 => Ok(Self::Paddle(Arc::new(Mutex::new(
                PaddleOCRVLQuantized::load(device).await?,
            )))),
        }
    }

    async fn run(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let model_name = match self {
            Self::Manga(_) => "manga-ocr",
            Self::Baberu(_) => "baberu-ocr",
            Self::Hayai(_) => "hayai-ocr",
            Self::Paddle(_) => "paddleocr-vl-1.6",
        };
        let page = input.page;
        let mut targets = Vec::new();
        let source = input
            .images
            .get(&input.scene, page, "source")
            .await?
            .ok_or_else(|| anyhow!("page {page} has no source image"))?;
        for entity in input.scene.descendants(page)? {
            let region = entity.id();
            if !input.contains_entity(region)? {
                continue;
            }
            let is_text_region = input
                .scene
                .component::<Region>(region)?
                .is_some_and(|value| value.kind == TextRegion::kind());
            if !is_text_region {
                continue;
            }
            let geometry = input
                .scene
                .component::<Geometry>(region)?
                .ok_or_else(|| anyhow!("text region {region} has no geometry"))?;
            let crop = crop(&source, &geometry)
                .with_context(|| format!("text region {region} is outside its source image"))?;
            for relation in input.scene.relations_to_as::<RecognizedFrom>(region) {
                let content = relation.value().source;
                let previous = input.scene.component::<SourceText>(content)?;
                if previous
                    .as_ref()
                    .is_some_and(|value| matches!(value.text.origin, Origin::User))
                {
                    continue;
                }
                targets.push(OcrTarget {
                    content,
                    region,
                    geometry: geometry.clone(),
                    previous,
                    image: crop.clone(),
                });
            }
        }

        let stop = input.stop.clone();
        let results = match self {
            Self::Manga(model) => {
                infer_text(model.clone(), targets, stop, |model, image| {
                    model.inference(image)
                })
                .await?
            }
            Self::Baberu(model) => {
                infer_text(model.clone(), targets, stop, |model, image| {
                    model.inference(image)
                })
                .await?
            }
            Self::Hayai(model) => {
                infer_text(model.clone(), targets, stop, |model, image| {
                    model.inference(image)
                })
                .await?
            }
            Self::Paddle(model) => {
                infer_text(model.clone(), targets, stop, |model, image| {
                    Ok(model.inference(image, PaddleOCRVLTask::Ocr)?.text)
                })
                .await?
            }
        };

        let generation = generation(PRODUCER, model_name)?;
        let mut edit = input.scene.edit_as(generation.clone());
        edit.observe_assets(page)?;
        for result in &results {
            edit.observe::<Region>(result.region)?;
            edit.observe::<Geometry>(result.region)?;
            edit.observe::<SourceText>(result.content)?;
        }
        for result in results {
            let language = result
                .previous
                .and_then(|value| value.language)
                .or_else(|| LanguageTag::new("ja-JP").ok());
            edit.set(
                result.content,
                &SourceText {
                    text: Authored::generated(result.text, generation.clone()),
                    language,
                },
            )?;
            let (min_x, min_y, max_x, max_y) = geometry_extents(&result.geometry)
                .ok_or_else(|| anyhow!("text region {} has empty geometry", result.region))?;
            edit.set(
                result.region,
                &OcrAnalysis {
                    origin: Origin::Generated(generation.clone()),
                    direction: if max_y - min_y >= (max_x - min_x) * 1.15 {
                        TextDirection::Vertical
                    } else {
                        TextDirection::Horizontal
                    },
                    confidence: None,
                    line_boundaries: Vec::new(),
                },
            )?;
        }
        finish(edit)
    }
}

struct OcrTarget {
    content: EntityId,
    region: EntityId,
    geometry: Geometry,
    previous: Option<SourceText>,
    image: DynamicImage,
}

struct OcrResult {
    content: EntityId,
    region: EntityId,
    geometry: Geometry,
    previous: Option<SourceText>,
    text: String,
}

async fn infer_text<M: Send + 'static>(
    model: Arc<Mutex<M>>,
    targets: Vec<OcrTarget>,
    stop: StopToken,
    inference: impl Fn(&M, &DynamicImage) -> Result<String> + Send + Sync + 'static,
) -> Result<Vec<OcrResult>> {
    tokio_rayon::spawn(move || {
        let model = model
            .lock()
            .map_err(|_| anyhow!("OCR model lock is poisoned"))?;
        targets
            .into_iter()
            .map(|target| {
                if stop.stopped() {
                    bail!("OCR was stopped");
                }
                Ok(OcrResult {
                    content: target.content,
                    region: target.region,
                    geometry: target.geometry,
                    previous: target.previous,
                    text: normalize_ocr_text(inference(&model, &target.image)?),
                })
            })
            .collect()
    })
    .await
}

// Manga OCR can emit replacement-box glyphs for an isolated Japanese ellipsis.
// Normalize only an all-placeholder sequence so ordinary OCR output is preserved.
fn normalize_ocr_text(text: String) -> String {
    let visible = text
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<Vec<_>>();
    if visible.len() >= 2
        && visible
            .iter()
            .all(|character| matches!(character, '☐' | '□' | '▢' | '▣' | '�'))
    {
        "…".to_owned()
    } else {
        text
    }
}

fn crop(source: &DynamicImage, geometry: &Geometry) -> Result<DynamicImage> {
    let (min_x, min_y, max_x, max_y) =
        geometry_extents(geometry).ok_or_else(|| anyhow!("geometry is empty"))?;
    let x = min_x.floor().max(0.0) as u32;
    let y = min_y.floor().max(0.0) as u32;
    let right = max_x.ceil().max(0.0).min(f64::from(source.width())) as u32;
    let bottom = max_y.ceil().max(0.0).min(f64::from(source.height())) as u32;
    if right <= x || bottom <= y {
        bail!("geometry does not overlap the image");
    }
    Ok(source.crop_imm(x, y, right - x, bottom - y))
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use image::DynamicImage;
    use koharu_scene::{EntityId, Geometry};

    use super::{OcrTarget, infer_text, normalize_ocr_text};
    use crate::StopToken;

    #[tokio::test]
    async fn stop_ends_the_region_loop() {
        let stop = StopToken::default();
        let calls = Arc::new(AtomicUsize::new(0));
        let targets = (0..3)
            .map(|_| OcrTarget {
                content: EntityId::new(),
                region: EntityId::new(),
                geometry: Geometry::rectangle(0.0, 0.0, 1.0, 1.0),
                previous: None,
                image: DynamicImage::new_luma8(1, 1),
            })
            .collect();
        let inference = {
            let stop = stop.clone();
            let calls = calls.clone();
            move |_: &(), _: &DynamicImage| {
                calls.fetch_add(1, Ordering::SeqCst);
                stop.stop();
                Ok(String::new())
            }
        };

        let result = infer_text(Arc::new(Mutex::new(())), targets, stop, inference).await;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "OCR kept running after stop"
        );
        assert!(
            result.is_err(),
            "a stopped OCR loop must not return partial text"
        );
    }

    #[test]
    fn repeated_placeholder_glyphs_are_an_ellipsis() {
        assert_eq!(normalize_ocr_text("☐ ☐ ☐".to_owned()), "…");
        assert_eq!(normalize_ocr_text("□\n□".to_owned()), "…");
    }

    #[test]
    fn ordinary_text_and_single_boxes_are_unchanged() {
        assert_eq!(normalize_ocr_text("待って…".to_owned()), "待って…");
        assert_eq!(normalize_ocr_text("☐".to_owned()), "☐");
    }
}
